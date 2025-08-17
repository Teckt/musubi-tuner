"""
Gradio UI for Musubi Tuner WAN 2.1/2.2 LoRA Training

This UI provides an intuitive interface for training LoRA models with WAN 2.1 and WAN 2.2 architectures.
Features:
- Model configuration with task-specific settings
- Dataset configuration with TOML generation
- Training configuration with memory optimization options
- Config save/load functionality
- Training command generation and execution
- Caption generation for training datasets
"""

import gradio as gr
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from config_manager import ConfigManager

# Import for caption generation
try:
    from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch
    CAPTION_AVAILABLE = True
except ImportError:
    CAPTION_AVAILABLE = False

class MusubiTrainerUI:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.training_process = None
        self.training_thread = None
        self.is_training = False
        
        # Caption generation setup
        self.caption_model = None
        self.caption_processor = None
        self.caption_pipeline = None
    
    def get_training_status(self):
        """Get current training status"""
        if self.is_training and self.training_process:
            if self.training_process.poll() is None:
                return "Training in progress..."
            else:
                # Process finished
                self.is_training = False
                return_code = self.training_process.returncode
                if return_code == 0:
                    return "Training completed successfully"
                else:
                    return f"Training failed with exit code {return_code}"
        elif self.is_training:
            return "Training starting..."
        else:
            return "Ready"
    
    def cleanup(self):
        """Cleanup resources when UI is closed"""
        if self.is_training and self.training_process:
            print("DEBUG: Cleaning up training process...")
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
            except:
                try:
                    self.training_process.kill()
                except:
                    pass
            self.is_training = False
            self.training_process = None
        
    def create_model_config_ui(self):
        """Create model configuration UI"""
        with gr.Group():
            gr.Markdown("## Model Configuration")
            
            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=[(info["name"], task) for task, info in self.config_manager.wan_tasks.items()],
                    label="Task",
                    value="t2v-14B",
                    info="Select the WAN model task to train"
                )
                
            with gr.Group():
                gr.Markdown("### Model Paths")
                
                dit_path = gr.Textbox(
                    label="DiT Model Path",
                    placeholder="path/to/wan2.x_model.safetensors",
                    info="Main DiT model weights file"
                )
                
                dit_high_noise_path = gr.Textbox(
                    label="DiT High Noise Model Path (WAN 2.2 only)",
                    placeholder="path/to/wan2.2_high_noise_model.safetensors",
                    info="High noise model for WAN 2.2 dual model training",
                    visible=False
                )
                
                t5_path = gr.Textbox(
                    label="T5 Model Path",
                    placeholder="path/to/models_t5_umt5-xxl-enc-bf16.pth",
                    info="T5 text encoder model file"
                )
                
                clip_path = gr.Textbox(
                    label="CLIP Model Path",
                    placeholder="path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    info="CLIP text encoder (required for I2V tasks in WAN 2.1)",
                    visible=False
                )
                
                vae_path = gr.Textbox(
                    label="VAE Model Path",
                    placeholder="path/to/wan_2.1_vae.safetensors",
                    info="VAE model for encoding/decoding"
                )
                
            with gr.Group():
                gr.Markdown("### WAN 2.2 Specific Settings")
                
                with gr.Row():
                    timestep_boundary = gr.Number(
                        label="Timestep Boundary",
                        value=None,
                        precision=3,
                        info="Boundary for switching between high/low noise models (0.0-1.0 or 0-1000)",
                        visible=False
                    )
                    
                    offload_inactive_dit = gr.Checkbox(
                        label="Offload Inactive DiT",
                        value=False,
                        info="Offload inactive DiT model to CPU to save VRAM",
                        visible=False
                    )
                    
                    lazy_loading = gr.Checkbox(
                        label="Lazy Loading",
                        value=False,
                        info="Enable lazy loading for DiT models to save VRAM",
                        visible=False
                    )
        
        def update_model_ui(task):
            """Update UI visibility based on selected task"""
            task_info = self.config_manager.get_task_info(task)
            is_wan22 = task_info.get("version") == "2.2"
            requires_clip = task_info.get("requires_clip", False)
            
            recommended = self.config_manager.get_recommended_settings_for_task(task)
            
            return {
                dit_high_noise_path: gr.Textbox(visible=is_wan22),
                clip_path: gr.Textbox(visible=requires_clip),
                timestep_boundary: gr.Number(
                    visible=is_wan22, 
                    value=recommended.get("timestep_boundary")
                ),
                offload_inactive_dit: gr.Checkbox(visible=is_wan22),
                lazy_loading: gr.Checkbox(visible=is_wan22)
            }
        
        task_dropdown.change(
            update_model_ui,
            inputs=[task_dropdown],
            outputs=[dit_high_noise_path, clip_path, timestep_boundary, 
                    offload_inactive_dit, lazy_loading]
        )
        
        return {
            "task": task_dropdown,
            "dit_path": dit_path,
            "dit_high_noise_path": dit_high_noise_path,
            "t5_path": t5_path,
            "clip_path": clip_path,
            "vae_path": vae_path,
            "timestep_boundary": timestep_boundary,
            "offload_inactive_dit": offload_inactive_dit,
            "lazy_loading": lazy_loading
        }
    
    def create_dataset_config_ui(self):
        """Create dataset configuration UI"""
        with gr.Group():
            gr.Markdown("## Dataset Configuration")
            
            with gr.Group():
                gr.Markdown("### General Settings")
                
                with gr.Row():
                    resolution_width = gr.Number(
                        label="Width",
                        value=960,
                        precision=0,
                        info="Training resolution width"
                    )
                    resolution_height = gr.Number(
                        label="Height", 
                        value=544,
                        precision=0,
                        info="Training resolution height"
                    )
                    
                with gr.Row():
                    caption_extension = gr.Textbox(
                        label="Caption Extension",
                        value=".txt",
                        info="File extension for caption files"
                    )
                    
                    dataset_batch_size = gr.Number(
                        label="Dataset Batch Size",
                        value=1,
                        precision=0,
                        info="Batch size for dataset loading"
                    )
                    
                with gr.Row():
                    enable_bucket = gr.Checkbox(
                        label="Enable Bucketing",
                        value=True,
                        info="Enable aspect ratio bucketing"
                    )
                    
                    bucket_no_upscale = gr.Checkbox(
                        label="No Upscale",
                        value=False,
                        info="Disable upscaling in bucketing"
                    )
            
            with gr.Group():
                gr.Markdown("### Dataset Entries")
                
                # Image dataset
                with gr.Group():
                    gr.Markdown("#### Image Dataset")
                    
                    image_directory = gr.Textbox(
                        label="Image Directory",
                        placeholder="path/to/image_directory",
                        info="Directory containing training images"
                    )
                    
                    image_cache_directory = gr.Textbox(
                        label="Image Cache Directory",
                        placeholder="path/to/image_cache",
                        info="Directory for caching image latents"
                    )
                    
                    image_num_repeats = gr.Number(
                        label="Number of Repeats",
                        value=1,
                        precision=0,
                        info="How many times to repeat this dataset"
                    )
                
                # Video dataset
                with gr.Group():
                    gr.Markdown("#### Video Dataset")
                    
                    video_directory = gr.Textbox(
                        label="Video Directory",
                        placeholder="path/to/video_directory",
                        info="Directory containing training videos"
                    )
                    
                    video_cache_directory = gr.Textbox(
                        label="Video Cache Directory", 
                        placeholder="path/to/video_cache",
                        info="Directory for caching video latents"
                    )
                    
                    target_frames = gr.Textbox(
                        label="Target Frames",
                        value="1,25,45",
                        info="Comma-separated list of frame counts (must be N*4+1)"
                    )
                    
                    frame_extraction = gr.Dropdown(
                        choices=["head", "chunk", "slide", "uniform", "full"],
                        label="Frame Extraction Method",
                        value="head",
                        info="Method for extracting frames from videos"
                    )
                    
                    with gr.Row():
                        frame_stride = gr.Number(
                            label="Frame Stride",
                            value=1,
                            precision=0,
                            info="Stride for slide extraction"
                        )
                        
                        frame_sample = gr.Number(
                            label="Frame Sample",
                            value=4,
                            precision=0,
                            info="Number of samples for uniform extraction"
                        )
                        
                        max_frames = gr.Number(
                            label="Max Frames",
                            value=129,
                            precision=0,
                            info="Maximum frames for full extraction"
                        )
                    
                    source_fps = gr.Number(
                        label="Source FPS",
                        value=None,
                        precision=1,
                        info="Source video FPS (use decimal, e.g., 30.0)"
                    )
                    
                    video_num_repeats = gr.Number(
                        label="Number of Repeats",
                        value=1,
                        precision=0,
                        info="How many times to repeat this dataset"
                    )
            
            # TOML generation
            with gr.Group():
                gr.Markdown("### Dataset TOML Generation")
                
                toml_output_path = gr.Textbox(
                    label="TOML Output Path",
                    placeholder="dataset_config.toml",
                    info="Path where to save the dataset configuration TOML file"
                )
                
                generate_toml_btn = gr.Button("Generate Dataset TOML", variant="primary")
                toml_status = gr.Textbox(label="TOML Generation Status", interactive=False)
        
        def generate_dataset_toml(*args):
            """Generate dataset TOML configuration"""
            try:
                (res_w, res_h, cap_ext, ds_batch, enable_buck, bucket_no_up,
                 img_dir, img_cache, img_repeats, vid_dir, vid_cache, 
                 tgt_frames, frame_ext, frame_stride_val, frame_sample_val, 
                 max_frames_val, src_fps, vid_repeats, toml_path) = args
                
                config = {
                    "general": {
                        "resolution": [int(res_w), int(res_h)],
                        "caption_extension": cap_ext,
                        "batch_size": int(ds_batch),
                        "enable_bucket": enable_buck,
                        "bucket_no_upscale": bucket_no_up
                    },
                    "datasets": []
                }
                
                # Add image dataset if directory specified
                if img_dir.strip():
                    img_dataset = {
                        "image_directory": img_dir.strip(),
                        "num_repeats": int(img_repeats)
                    }
                    if img_cache.strip():
                        img_dataset["cache_directory"] = img_cache.strip()
                    config["datasets"].append(img_dataset)
                
                # Add video dataset if directory specified
                if vid_dir.strip():
                    vid_dataset = {
                        "video_directory": vid_dir.strip(),
                        "target_frames": [int(x.strip()) for x in tgt_frames.split(",") if x.strip()],
                        "frame_extraction": frame_ext,
                        "num_repeats": int(vid_repeats)
                    }
                    
                    if vid_cache.strip():
                        vid_dataset["cache_directory"] = vid_cache.strip()
                    if frame_stride_val and frame_ext == "slide":
                        vid_dataset["frame_stride"] = int(frame_stride_val)
                    if frame_sample_val and frame_ext == "uniform":
                        vid_dataset["frame_sample"] = int(frame_sample_val)
                    if max_frames_val and frame_ext == "full":
                        vid_dataset["max_frames"] = int(max_frames_val)
                    if src_fps:
                        vid_dataset["source_fps"] = float(src_fps)
                        
                    config["datasets"].append(vid_dataset)
                
                if not config["datasets"]:
                    return "Error: No datasets specified"
                
                # Generate TOML file
                output_path = toml_path.strip() or "dataset_config.toml"
                self.config_manager.generate_dataset_toml(config, output_path)
                
                return f"Successfully generated dataset TOML at: {output_path}"
                
            except Exception as e:
                return f"Error generating TOML: {str(e)}"
        
        generate_toml_btn.click(
            generate_dataset_toml,
            inputs=[
                resolution_width, resolution_height, caption_extension, dataset_batch_size,
                enable_bucket, bucket_no_upscale, image_directory, image_cache_directory,
                image_num_repeats, video_directory, video_cache_directory, target_frames,
                frame_extraction, frame_stride, frame_sample, max_frames, source_fps,
                video_num_repeats, toml_output_path
            ],
            outputs=[toml_status]
        )
        
        return {
            "resolution_width": resolution_width,
            "resolution_height": resolution_height,
            "caption_extension": caption_extension,
            "dataset_batch_size": dataset_batch_size,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale,
            "image_directory": image_directory,
            "image_cache_directory": image_cache_directory,
            "image_num_repeats": image_num_repeats,
            "video_directory": video_directory,
            "video_cache_directory": video_cache_directory,
            "target_frames": target_frames,
            "frame_extraction": frame_extraction,
            "frame_stride": frame_stride,
            "frame_sample": frame_sample,
            "max_frames": max_frames,
            "source_fps": source_fps,
            "video_num_repeats": video_num_repeats,
            "toml_output_path": toml_output_path,
            "toml_status": toml_status
        }
    
    def create_training_config_ui(self):
        """Create training configuration UI"""
        with gr.Group():
            gr.Markdown("## Training Configuration")
            
            with gr.Group():
                gr.Markdown("### Basic Training Settings")
                
                with gr.Row():
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=2e-4,
                        precision=6,
                        info="Learning rate for training"
                    )
                    
                    optimizer_type = gr.Dropdown(
                        choices=self.config_manager.optimizer_types,
                        label="Optimizer",
                        value="adamw8bit",
                        info="Optimizer type"
                    )
                
                with gr.Row():
                    network_dim = gr.Number(
                        label="Network Dimension",
                        value=32,
                        precision=0,
                        info="LoRA network dimension"
                    )
                    
                    network_alpha = gr.Number(
                        label="Network Alpha",
                        value=None,
                        precision=0,
                        info="LoRA network alpha (leave empty for auto)"
                    )
                
                with gr.Row():
                    max_train_epochs = gr.Number(
                        label="Max Train Epochs",
                        value=16,
                        precision=0,
                        info="Maximum number of training epochs"
                    )
                    
                    save_every_n_epochs = gr.Number(
                        label="Save Every N Epochs",
                        value=1,
                        precision=0,
                        info="Save checkpoint every N epochs"
                    )
                
                with gr.Row():
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        precision=0,
                        info="Training batch size"
                    )
                    
                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                        info="Random seed for reproducibility"
                    )
            
            with gr.Group():
                gr.Markdown("### Advanced Training Settings")
                
                with gr.Row():
                    timestep_sampling = gr.Dropdown(
                        choices=self.config_manager.timestep_sampling_modes,
                        label="Timestep Sampling",
                        value="shift",
                        info="Timestep sampling method"
                    )
                    
                    discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=3.0,
                        precision=1,
                        info="Flow shift value for training"
                    )
                
                with gr.Row():
                    min_timestep = gr.Number(
                        label="Min Timestep",
                        value=None,
                        precision=0,
                        info="Minimum timestep (0-1000, leave empty for default)"
                    )
                    
                    max_timestep = gr.Number(
                        label="Max Timestep", 
                        value=None,
                        precision=0,
                        info="Maximum timestep (0-1000, leave empty for default)"
                    )
                
                preserve_distribution_shape = gr.Checkbox(
                    label="Preserve Distribution Shape",
                    value=False,
                    info="Maintain timestep distribution shape"
                )
            
            with gr.Group():
                gr.Markdown("### Memory & Performance Settings")
                
                with gr.Row():
                    mixed_precision = gr.Dropdown(
                        choices=["no", "fp16", "bf16"],
                        label="Mixed Precision",
                        value="bf16",
                        info="Mixed precision training mode"
                    )
                    
                    blocks_to_swap = gr.Number(
                        label="Blocks to Swap",
                        value=None,
                        precision=0,
                        info="Number of blocks to swap to CPU (max 39 for 14B, 29 for 1.3B)"
                    )
                
                with gr.Row():
                    fp8_base = gr.Checkbox(
                        label="FP8 Base",
                        value=True,
                        info="Use FP8 for base model"
                    )
                    
                    fp8_scaled = gr.Checkbox(
                        label="FP8 Scaled",
                        value=False,
                        info="Use scaled FP8 optimization"
                    )
                    
                    fp8_t5 = gr.Checkbox(
                        label="FP8 T5",
                        value=False,
                        info="Use FP8 for T5 text encoder"
                    )
                
                with gr.Row():
                    gradient_checkpointing = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=True,
                        info="Enable gradient checkpointing"
                    )
                    
                    vae_cache_cpu = gr.Checkbox(
                        label="VAE Cache CPU",
                        value=False,
                        info="Cache VAE features on CPU"
                    )
                
                with gr.Row():
                    max_data_loader_n_workers = gr.Number(
                        label="Data Loader Workers",
                        value=2,
                        precision=0,
                        info="Number of data loader workers"
                    )
                    
                    persistent_data_loader_workers = gr.Checkbox(
                        label="Persistent Data Loader Workers",
                        value=True,
                        info="Keep data loader workers persistent"
                    )
            
            with gr.Group():
                gr.Markdown("### Attention & Memory Optimization")
                
                attention_mode = gr.Radio(
                    choices=self.config_manager.memory_attention_modes,
                    label="Attention Mode",
                    value="sdpa",
                    info="Attention implementation to use"
                )
                
                split_attn = gr.Checkbox(
                    label="Split Attention",
                    value=False,
                    info="Process attention in chunks to save memory"
                )
            
            with gr.Group():
                gr.Markdown("### Output Settings")
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="outputs",
                    info="Directory to save training outputs"
                )
                
                output_name = gr.Textbox(
                    label="Output Name",
                    value="wan_lora",
                    info="Base name for output files"
                )
                
                dataset_config_path = gr.Textbox(
                    label="Dataset Config Path",
                    placeholder="dataset_config.toml",
                    info="Path to dataset configuration TOML file"
                )
            
            with gr.Group():
                gr.Markdown("### Logging & Monitoring")
                
                with gr.Row():
                    logging_dir = gr.Textbox(
                        label="Logging Directory",
                        value="",
                        info="Directory for logs (leave empty to disable)"
                    )
                    
                    log_with = gr.Dropdown(
                        choices=["", "tensorboard", "wandb"],
                        label="Log With",
                        value="",
                        info="Logging service to use"
                    )
                
                with gr.Row():
                    wandb_api_key = gr.Textbox(
                        label="WandB API Key",
                        value="",
                        type="password",
                        info="WandB API key (if using WandB logging)"
                    )
                    
                    wandb_run_name = gr.Textbox(
                        label="WandB Run Name",
                        value="",
                        info="Name for WandB run"
                    )
                
                log_config = gr.Checkbox(
                    label="Log Config",
                    value=False,
                    info="Log training configuration"
                )
                
                debug_swapping = gr.Checkbox(
                    label="Debug Model Swapping",
                    value=False,
                    info="Enable detailed logging of model swapping operations (CPU/GPU transfers)"
                )
        
        return {
            "learning_rate": learning_rate,
            "optimizer_type": optimizer_type,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "max_train_epochs": max_train_epochs,
            "save_every_n_epochs": save_every_n_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "timestep_sampling": timestep_sampling,
            "discrete_flow_shift": discrete_flow_shift,
            "min_timestep": min_timestep,
            "max_timestep": max_timestep,
            "preserve_distribution_shape": preserve_distribution_shape,
            "mixed_precision": mixed_precision,
            "blocks_to_swap": blocks_to_swap,
            "fp8_base": fp8_base,
            "fp8_scaled": fp8_scaled,
            "fp8_t5": fp8_t5,
            "gradient_checkpointing": gradient_checkpointing,
            "vae_cache_cpu": vae_cache_cpu,
            "max_data_loader_n_workers": max_data_loader_n_workers,
            "persistent_data_loader_workers": persistent_data_loader_workers,
            "attention_mode": attention_mode,
            "split_attn": split_attn,
            "output_dir": output_dir,
            "output_name": output_name,
            "dataset_config_path": dataset_config_path,
            "logging_dir": logging_dir,
            "log_with": log_with,
            "wandb_api_key": wandb_api_key,
            "wandb_run_name": wandb_run_name,
            "log_config": log_config,
            "debug_swapping": debug_swapping
        }
    
    def create_config_management_ui(self):
        """Create configuration save/load UI"""
        with gr.Group():
            gr.Markdown("## Configuration Management")
            
            with gr.Row():
                config_name = gr.Textbox(
                    label="Configuration Name",
                    placeholder="my_wan_config",
                    info="Name for saving/loading configurations"
                )
                
                config_list = gr.Dropdown(
                    choices=self.config_manager.list_configs(),
                    label="Saved Configurations",
                    info="Select a saved configuration to load"
                )
            
            with gr.Row():
                save_config_btn = gr.Button("Save Configuration", variant="primary")
                load_config_btn = gr.Button("Load Configuration")
                delete_config_btn = gr.Button("Delete Configuration", variant="stop")
                refresh_configs_btn = gr.Button("Refresh List")
            
            config_status = gr.Textbox(label="Configuration Status", interactive=False)
            
        return {
            "config_name": config_name,
            "config_list": config_list,
            "save_config_btn": save_config_btn,
            "load_config_btn": load_config_btn,
            "delete_config_btn": delete_config_btn,
            "refresh_configs_btn": refresh_configs_btn,
            "config_status": config_status
        }
    
    def create_training_execution_ui(self):
        """Create training execution UI"""
        with gr.Group():
            gr.Markdown("## Training Execution")
            
            with gr.Group():
                gr.Markdown("### Pre-training Steps")
                gr.Markdown("**Note**: WAN cache scripts have different options than HunyuanVideo. VAE chunk size and tiling are not supported for WAN caching.")
                
                cache_latents_btn = gr.Button("Cache Latents", variant="secondary")
                cache_text_encoder_btn = gr.Button("Cache Text Encoder Outputs", variant="secondary")
                
                with gr.Row():
                    text_encoder_batch_size = gr.Number(
                        label="Text Encoder Batch Size",
                        value=16,
                        precision=0,
                        info="Batch size for text encoder caching"
                    )
            
            with gr.Group():
                gr.Markdown("### Training Command")
                
                generated_command = gr.Textbox(
                    label="Generated Training Command",
                    lines=10,
                    interactive=False,
                    info="Generated command for training"
                )
                
                generate_command_btn = gr.Button("Generate Training Command", variant="primary")
                
            with gr.Group():
                gr.Markdown("### Training Execution")
                
                with gr.Row():
                    start_training_btn = gr.Button("Start Training", variant="primary")
                    stop_training_btn = gr.Button("Stop Training", variant="stop")
                    refresh_status_btn = gr.Button("Refresh Status", variant="secondary")
                
                training_status = gr.Textbox(
                    label="Training Status",
                    value="Ready",
                    interactive=False
                )
                
                with gr.Row():
                    gr.Markdown("**Important**: Training output will appear in the terminal/console. Use the buttons above to control training.")
                
                training_output = gr.Textbox(
                    label="Training Log",
                    lines=15,
                    interactive=False,
                    info="Recent training messages and status updates"
                )
        
        return {
            "cache_latents_btn": cache_latents_btn,
            "cache_text_encoder_btn": cache_text_encoder_btn,
            "text_encoder_batch_size": text_encoder_batch_size,
            "generated_command": generated_command,
            "generate_command_btn": generate_command_btn,
            "start_training_btn": start_training_btn,
            "stop_training_btn": stop_training_btn,
            "refresh_status_btn": refresh_status_btn,
            "training_status": training_status,
            "training_output": training_output
        }
    
    def build_training_command(self, model_config, training_config):
        """Build the training command from configuration"""
        try:
            # run in venv
            accelerate_venv = ".venv/Scripts/accelerate.exe"
            cmd = [accelerate_venv, "launch", "--num_cpu_threads_per_process", "1", 
                   "--mixed_precision", training_config["mixed_precision"]]
            
            # Add the main script
            script_path = "src/musubi_tuner/wan_train_network.py"
            cmd.append(script_path)
            
            # Model arguments
            cmd.extend(["--task", model_config["task"]])
            cmd.extend(["--dit", model_config["dit_path"]])
            
            if model_config.get("dit_high_noise_path"):
                cmd.extend(["--dit_high_noise", model_config["dit_high_noise_path"]])
            
            cmd.extend(["--t5", model_config["t5_path"]])
            cmd.extend(["--vae", model_config["vae_path"]])
            
            if model_config.get("clip_path"):
                cmd.extend(["--clip", model_config["clip_path"]])
            
            # Training arguments
            cmd.extend(["--dataset_config", training_config["dataset_config_path"]])
            cmd.extend(["--network_module", "networks.lora_wan"])
            cmd.extend(["--network_dim", str(training_config["network_dim"])])
            
            if training_config.get("network_alpha") is not None and training_config["network_alpha"] > 0:
                cmd.extend(["--network_alpha", str(training_config["network_alpha"])])
            
            cmd.extend(["--learning_rate", str(training_config["learning_rate"])])
            cmd.extend(["--optimizer_type", training_config["optimizer_type"]])
            cmd.extend(["--max_train_epochs", str(training_config["max_train_epochs"])])
            cmd.extend(["--save_every_n_epochs", str(training_config["save_every_n_epochs"])])
            cmd.extend(["--seed", str(training_config["seed"])])
            cmd.extend(["--output_dir", training_config["output_dir"]])
            cmd.extend(["--output_name", training_config["output_name"]])
            
            # Memory and performance settings
            if training_config["fp8_base"]:
                cmd.append("--fp8_base")
            
            if training_config["fp8_scaled"]:
                cmd.append("--fp8_scaled")
            
            if training_config["fp8_t5"]:
                cmd.append("--fp8_t5")
            
            if training_config["gradient_checkpointing"]:
                cmd.append("--gradient_checkpointing")
            
            if training_config["vae_cache_cpu"]:
                cmd.append("--vae_cache_cpu")
                
            if training_config.get("blocks_to_swap") is not None:
                cmd.extend(["--blocks_to_swap", str(int(training_config["blocks_to_swap"]))])
            
            # Attention mode
            attention_mode = training_config["attention_mode"]
            if attention_mode == "sdpa":
                cmd.append("--sdpa")
            elif attention_mode == "flash_attn":
                cmd.append("--flash_attn")
            elif attention_mode == "xformers":
                cmd.append("--xformers")
            elif attention_mode == "sage_attn":
                cmd.append("--sage_attn")
            
            if training_config["split_attn"]:
                cmd.append("--split_attn")
            
            # Timestep settings
            cmd.extend(["--timestep_sampling", training_config["timestep_sampling"]])
            cmd.extend(["--discrete_flow_shift", str(training_config["discrete_flow_shift"])])
            
            if training_config.get("min_timestep") is not None and training_config["min_timestep"] > 0:
                cmd.extend(["--min_timestep", str(int(training_config["min_timestep"]))])
            
            if training_config.get("max_timestep") is not None and training_config["max_timestep"] > 0:
                cmd.extend(["--max_timestep", str(int(training_config["max_timestep"]))])
            
            if training_config["preserve_distribution_shape"]:
                cmd.append("--preserve_distribution_shape")
            
            # WAN 2.2 specific settings
            if model_config.get("timestep_boundary") is not None:
                # Convert float timestep boundary (0.0-1.0) to integer (0-1000)
                boundary_val = model_config["timestep_boundary"]
                if isinstance(boundary_val, float) and 0.0 <= boundary_val <= 1.0:
                    boundary_int = int(boundary_val * 1000)
                    cmd.extend(["--timestep_boundary", str(boundary_int)])
                else:
                    # Already an integer, use as-is
                    cmd.extend(["--timestep_boundary", str(int(boundary_val))])
            
            if model_config.get("offload_inactive_dit"):
                cmd.append("--offload_inactive_dit")
            
            # Data loader settings
            cmd.extend(["--max_data_loader_n_workers", str(training_config["max_data_loader_n_workers"])])
            
            if training_config["persistent_data_loader_workers"]:
                cmd.append("--persistent_data_loader_workers")
            
            # Logging settings
            if training_config.get("logging_dir"):
                cmd.extend(["--logging_dir", training_config["logging_dir"]])
            
            if training_config.get("log_with"):
                cmd.extend(["--log_with", training_config["log_with"]])
            
            if training_config.get("wandb_api_key"):
                cmd.extend(["--wandb_api_key", training_config["wandb_api_key"]])
            
            if training_config.get("wandb_run_name"):
                cmd.extend(["--wandb_run_name", training_config["wandb_run_name"]])
            
            if training_config["log_config"]:
                cmd.append("--log_config")
            
            if training_config["debug_swapping"]:
                cmd.append("--debug_swapping")
            
            return " ".join(cmd)
            
        except Exception as e:
            return f"Error building command: {str(e)}"
    
    def create_interface(self):
        """Create the main Gradio interface"""
        with gr.Blocks(title="Musubi Tuner - WAN 2.1/2.2 LoRA Training", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # Musubi Tuner - WAN 2.1/2.2 LoRA Training
            
            This interface provides an easy way to train LoRA models for WAN 2.1 and WAN 2.2 video generation models.
            
            ## Quick Start:
            1. **Model Configuration**: Select your task and provide model paths
            2. **Dataset Configuration**: Set up your training data and generate TOML config
            3. **Training Configuration**: Adjust training parameters and memory settings
            4. **Execute**: Generate commands and start training
            
            **Important**: Make sure to cache latents and text encoder outputs before training!
            """)
            
            with gr.Tabs():
                with gr.Tab("Model Configuration"):
                    model_components = self.create_model_config_ui()
                
                with gr.Tab("Dataset Configuration"):
                    dataset_components = self.create_dataset_config_ui()
                
                with gr.Tab("Training Configuration"):
                    training_components = self.create_training_config_ui()
                
                with gr.Tab("Configuration Management"):
                    config_components = self.create_config_management_ui()
                
                # Add caption generator tab
                self.create_caption_interface()
                
                with gr.Tab("Training Execution"):
                    execution_components = self.create_training_execution_ui()
            
            # Setup event handlers for configuration management
            def save_configuration(config_name, *args):
                """Save current configuration"""
                print(f"DEBUG: save_configuration called with config_name='{config_name}', args_count={len(args)}")
                
                if not config_name.strip():
                    return "Error: Configuration name is required", gr.Dropdown(choices=self.config_manager.list_configs())
                
                try:
                    # Debug: Print first few args to see what we're getting
                    print(f"DEBUG: First 10 args: {args[:10] if len(args) >= 10 else args}")
                    
                    # Extract all values and build config
                    config = self._extract_full_config(*args)
                    print(f"DEBUG: Extracted config structure: {json.dumps(config, indent=2, default=str)}")
                    
                    filename = f"{config_name.strip()}.json"
                    self.config_manager.save_config(config, filename)
                    print(f"DEBUG: Successfully saved config to {filename}")
                    
                    return f"Configuration saved as {filename}", gr.Dropdown(choices=self.config_manager.list_configs())
                except Exception as e:
                    print(f"DEBUG: Error in save_configuration: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return f"Error saving configuration: {str(e)}", gr.Dropdown(choices=self.config_manager.list_configs())
            
            def load_configuration(config_name):
                """Load configuration and return updates for all components"""
                print(f"DEBUG: load_configuration called with config_name='{config_name}'")
                
                if not config_name:
                    print("DEBUG: No configuration name provided")
                    return ["No configuration selected"] + [gr.update() for _ in range(50)]
                
                try:
                    filename = f"{config_name}.json"
                    print(f"DEBUG: Attempting to load config from {filename}")
                    
                    config = self.config_manager.load_config(filename)
                    if not config:
                        print("DEBUG: Config not found or failed to load")
                        return ["Configuration not found"] + [gr.update() for _ in range(50)]
                    
                    print(f"DEBUG: Loaded config: {json.dumps(config, indent=2, default=str)}")
                    
                    # Return status and all component updates
                    updates = [f"Configuration {config_name} loaded successfully"]
                    component_updates = self._config_to_component_updates(config)
                    print(f"DEBUG: Generated {len(component_updates)} component updates")
                    updates.extend(component_updates)
                    return updates
                except Exception as e:
                    print(f"DEBUG: Error in load_configuration: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return [f"Error loading configuration: {str(e)}"] + [gr.update() for _ in range(50)]
            
            def delete_configuration(config_name):
                """Delete selected configuration"""
                if not config_name:
                    return "No configuration selected", gr.Dropdown(choices=self.config_manager.list_configs())
                
                try:
                    if self.config_manager.delete_config(config_name):
                        return f"Configuration {config_name} deleted", gr.Dropdown(choices=self.config_manager.list_configs())
                    else:
                        return "Configuration not found", gr.Dropdown(choices=self.config_manager.list_configs())
                except Exception as e:
                    return f"Error deleting configuration: {str(e)}", gr.Dropdown(choices=self.config_manager.list_configs())
            
            def refresh_config_list():
                """Refresh the configuration list"""
                return gr.Dropdown(choices=self.config_manager.list_configs())
            
            def generate_training_command(*args):
                """Generate training command from current settings"""
                try:
                    config = self._extract_full_config(*args)
                    command = self.build_training_command(config["model"], config["training"])
                    return command
                except Exception as e:
                    return f"Error generating command: {str(e)}"
            
            def cache_latents_command(*args):
                """Generate and execute latent caching command"""
                print("DEBUG: cache_latents_command called")
                print(f"DEBUG: Received {len(args)} arguments")
                
                try:
                    print("DEBUG: Extracting configuration from arguments...")
                    config = self._extract_full_config(*args)
                    print("DEBUG: Configuration extracted successfully")
                    
                    model_config = config["model"]
                    execution_config = config["execution"]
                    training_config = config["training"]
                    
                    print(f"DEBUG: Model config - task: {model_config.get('task')}")
                    print(f"DEBUG: VAE path: {model_config.get('vae_path')}")
                    print(f"DEBUG: Dataset config path: {training_config.get('dataset_config_path')}")
                    
                    # Validate required paths
                    if not training_config.get("dataset_config_path"):
                        error_msg = "Error: Dataset config path is required for latent caching"
                        print(f"DEBUG: {error_msg}")
                        return error_msg
                    
                    if not model_config.get("vae_path"):
                        error_msg = "Error: VAE path is required for latent caching"
                        print(f"DEBUG: {error_msg}")
                        return error_msg
                    
                    print("DEBUG: Building command...")
                    venv_python = ".venv/scripts/python.exe"
                    cmd = [venv_python, "src/musubi_tuner/wan_cache_latents.py"]
                    cmd.extend(["--dataset_config", training_config["dataset_config_path"]])
                    cmd.extend(["--vae", model_config["vae_path"]])
                    
                    # WAN cache script doesn't support vae_chunk_size or vae_tiling
                    # These are only available in the HunyuanVideo cache script
                    print("DEBUG: Note - WAN cache script doesn't support --vae_chunk_size or --vae_tiling")
                    
                    # Check if I2V task for WAN 2.1 (requires --i2v flag)
                    task_info = self.config_manager.get_task_info(model_config["task"])
                    print(f"DEBUG: Task info: {task_info}")
                    
                    if "i2v" in model_config["task"].lower():
                        cmd.append("--i2v")
                        print("DEBUG: Added --i2v flag")
                        
                        # Add CLIP for WAN 2.1 I2V
                        if task_info.get("requires_clip") and model_config.get("clip_path"):
                            cmd.extend(["--clip", model_config["clip_path"]])
                            print(f"DEBUG: Added CLIP path: {model_config['clip_path']}")
                    
                    # Add memory optimization flags (this one is supported)
                    if training_config.get("vae_cache_cpu"):
                        cmd.append("--vae_cache_cpu")
                        print("DEBUG: Added VAE cache CPU")
                    
                    command_str = " ".join(cmd)
                    print(f"DEBUG: Final command: {command_str}")
                    
                    # Change to musubi-tuner directory to run the command
                    current_dir = Path(os.getcwd())
                    musubi_dir = current_dir.parent if current_dir.name == "ui" else current_dir
                    print(f"DEBUG: Current directory: {current_dir}")
                    print(f"DEBUG: Execution directory: {musubi_dir}")
                    
                    # Check if the script exists
                    script_path = musubi_dir / "src" / "musubi_tuner" / "wan_cache_latents.py"
                    print(f"DEBUG: Script path: {script_path}")
                    print(f"DEBUG: Script exists: {script_path.exists()}")
                    
                    if not script_path.exists():
                        error_msg = f"Error: Script not found at {script_path}"
                        print(f"DEBUG: {error_msg}")
                        return error_msg
                    
                    print("DEBUG: Executing command...")
                    # Execute the command
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(musubi_dir))
                    
                    print(f"DEBUG: Command completed with return code: {result.returncode}")
                    print(f"DEBUG: stdout length: {len(result.stdout) if result.stdout else 0}")
                    print(f"DEBUG: stderr length: {len(result.stderr) if result.stderr else 0}")
                    
                    if result.returncode == 0:
                        success_msg = f"Latent caching completed successfully!\n\nCommand: {command_str}\n\nOutput:\n{result.stdout}"
                        print("DEBUG: Command executed successfully")
                        return success_msg
                    else:
                        error_msg = f"Latent caching failed!\n\nCommand: {command_str}\n\nError:\n{result.stderr}"
                        print(f"DEBUG: Command failed with error: {result.stderr}")
                        return error_msg
                        
                except Exception as e:
                    error_msg = f"Error in latent caching: {str(e)}"
                    print(f"DEBUG: Exception occurred: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return error_msg
            
            def cache_text_encoder_command(*args):
                """Generate and execute text encoder caching command"""
                print("DEBUG: cache_text_encoder_command called")
                print(f"DEBUG: Received {len(args)} arguments")
                
                try:
                    print("DEBUG: Extracting configuration from arguments...")
                    config = self._extract_full_config(*args)
                    print("DEBUG: Configuration extracted successfully")
                    
                    model_config = config["model"]
                    execution_config = config["execution"]
                    training_config = config["training"]
                    
                    print(f"DEBUG: Model config - task: {model_config.get('task')}")
                    print(f"DEBUG: T5 path: {model_config.get('t5_path')}")
                    print(f"DEBUG: Dataset config path: {training_config.get('dataset_config_path')}")
                    
                    # Validate required paths
                    if not training_config.get("dataset_config_path"):
                        error_msg = "Error: Dataset config path is required for text encoder caching"
                        print(f"DEBUG: {error_msg}")
                        return error_msg
                    
                    if not model_config.get("t5_path"):
                        error_msg = "Error: T5 path is required for text encoder caching"
                        print(f"DEBUG: {error_msg}")
                        return error_msg
                    
                    print("DEBUG: Building command...")
                    venv_python = ".venv/scripts/python.exe"
                    cmd = [venv_python, "src/musubi_tuner/wan_cache_text_encoder_outputs.py"]
                    cmd.extend(["--dataset_config", training_config["dataset_config_path"]])
                    cmd.extend(["--t5", model_config["t5_path"]])
                    
                    # Add batch size (this is supported by the common parser)
                    if execution_config.get("text_encoder_batch_size"):
                        batch_size = str(int(execution_config["text_encoder_batch_size"]))
                        cmd.extend(["--batch_size", batch_size])
                        print(f"DEBUG: Added batch size: {batch_size}")
                    
                    # Add FP8 optimization (this is supported by WAN parser)
                    if training_config.get("fp8_t5"):
                        cmd.append("--fp8_t5")
                        print("DEBUG: Added fp8 T5")
                    
                    command_str = " ".join(cmd)
                    print(f"DEBUG: Final command: {command_str}")
                    
                    # Change to musubi-tuner directory to run the command
                    current_dir = Path(os.getcwd())
                    musubi_dir = current_dir.parent if current_dir.name == "ui" else current_dir
                    print(f"DEBUG: Current directory: {current_dir}")
                    print(f"DEBUG: Execution directory: {musubi_dir}")
                    
                    # Check if the script exists
                    script_path = musubi_dir / "src" / "musubi_tuner" / "wan_cache_text_encoder_outputs.py"
                    print(f"DEBUG: Script path: {script_path}")
                    print(f"DEBUG: Script exists: {script_path.exists()}")
                    
                    if not script_path.exists():
                        error_msg = f"Error: Script not found at {script_path}"
                        print(f"DEBUG: {error_msg}")
                        return error_msg
                    
                    print("DEBUG: Executing command...")
                    # Execute the command
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(musubi_dir))
                    
                    print(f"DEBUG: Command completed with return code: {result.returncode}")
                    print(f"DEBUG: stdout length: {len(result.stdout) if result.stdout else 0}")
                    print(f"DEBUG: stderr length: {len(result.stderr) if result.stderr else 0}")
                    
                    if result.returncode == 0:
                        success_msg = f"Text encoder caching completed successfully!\n\nCommand: {command_str}\n\nOutput:\n{result.stdout}"
                        print("DEBUG: Command executed successfully")
                        return success_msg
                    else:
                        error_msg = f"Text encoder caching failed!\n\nCommand: {command_str}\n\nError:\n{result.stderr}"
                        print(f"DEBUG: Command failed with error: {result.stderr}")
                        return error_msg
                        
                except Exception as e:
                    error_msg = f"Error in text encoder caching: {str(e)}"
                    print(f"DEBUG: Exception occurred: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return error_msg
            
            # Get all component inputs for event handlers
            # Only include input components, not buttons or output-only components
            all_components = []
            all_components.extend(model_components.values())
            
            # Add only input components from dataset 
            dataset_inputs = [
                dataset_components["resolution_width"],
                dataset_components["resolution_height"], 
                dataset_components["caption_extension"],
                dataset_components["dataset_batch_size"],
                dataset_components["enable_bucket"],
                dataset_components["bucket_no_upscale"],
                dataset_components["image_directory"],
                dataset_components["image_cache_directory"],
                dataset_components["image_num_repeats"],
                dataset_components["video_directory"],
                dataset_components["video_cache_directory"],
                dataset_components["target_frames"],
                dataset_components["frame_extraction"],
                dataset_components["frame_stride"],
                dataset_components["frame_sample"],
                dataset_components["max_frames"],
                dataset_components["source_fps"],
                dataset_components["video_num_repeats"],
                dataset_components["toml_output_path"]
            ]
            all_components.extend(dataset_inputs)
            
            all_components.extend(training_components.values())
            
            # Add only input components from execution (not buttons or outputs)
            execution_inputs = [
                execution_components["text_encoder_batch_size"]
            ]
            all_components.extend(execution_inputs)
            
            # Configuration management event handlers
            config_components["save_config_btn"].click(
                save_configuration,
                inputs=[config_components["config_name"]] + all_components,
                outputs=[config_components["config_status"], config_components["config_list"]]
            )
            
            config_components["load_config_btn"].click(
                load_configuration,
                inputs=[config_components["config_list"]],
                outputs=[config_components["config_status"]] + all_components
            )
            
            config_components["delete_config_btn"].click(
                delete_configuration,
                inputs=[config_components["config_list"]],
                outputs=[config_components["config_status"], config_components["config_list"]]
            )
            
            config_components["refresh_configs_btn"].click(
                refresh_config_list,
                outputs=[config_components["config_list"]]
            )
            
            print(f"DEBUG: Total components for event handlers: {len(all_components)}")
            print("DEBUG: Component types in all_components:")
            for i, component in enumerate(all_components):
                print(f"  {i}: {type(component).__name__}")
            
            # Training command generation
            execution_components["generate_command_btn"].click(
                generate_training_command,
                inputs=all_components,
                outputs=[execution_components["generated_command"]]
            )
            print("DEBUG: Generate command button event handler connected")
            
            # Cache commands
            execution_components["cache_latents_btn"].click(
                cache_latents_command,
                inputs=all_components,
                outputs=[execution_components["training_output"]]
            )
            print("DEBUG: Cache latents button event handler connected")
            
            execution_components["cache_text_encoder_btn"].click(
                cache_text_encoder_command,
                inputs=all_components,
                outputs=[execution_components["training_output"]]
            )
            print("DEBUG: Cache text encoder button event handler connected")
            
            # Training execution functions
            def start_training_function(*args):
                """Start training with the current configuration"""
                print("DEBUG: start_training_function called")
                
                if self.is_training:
                    return "Training is already running", "Training in progress..."
                
                try:
                    # Extract configuration and generate command
                    config = self._extract_full_config(*args)
                    command = self.build_training_command(config["model"], config["training"])
                    
                    if command.startswith("Error"):
                        return "Error generating training command", f"Failed to start training: {command}"
                    
                    # Start training in a separate thread
                    self.is_training = True
                    
                    def run_training():
                        """Run training in background thread"""
                        try:
                            print(f"DEBUG: Starting training with command: {command}")
                            
                            # Change to musubi-tuner directory
                            current_dir = Path(os.getcwd())
                            musubi_dir = current_dir.parent if current_dir.name == "ui" else current_dir
                            
                            # Parse command and execute
                            cmd_parts = command.split()
                            print(f"DEBUG: Command parts: {cmd_parts[:5]}...")  # Don't log full command for security
                            
                            # Start the training process
                            self.training_process = subprocess.Popen(
                                cmd_parts,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                cwd=str(musubi_dir),
                                bufsize=1,
                                universal_newlines=True
                            )
                            
                            print(f"DEBUG: Training process started with PID: {self.training_process.pid}")
                            
                            # Monitor the process output
                            output_lines = []
                            for line in iter(self.training_process.stdout.readline, ''):
                                if line:
                                    output_lines.append(line.rstrip())
                                    print(f"TRAINING: {line.rstrip()}")
                                    
                                    # Keep only last 100 lines to prevent memory issues
                                    if len(output_lines) > 100:
                                        output_lines = output_lines[-100:]
                                
                                # Check if process was terminated
                                if self.training_process.poll() is not None:
                                    break
                            
                            # Wait for process to complete
                            return_code = self.training_process.wait()
                            
                            if return_code == 0:
                                print("DEBUG: Training completed successfully")
                                final_output = "\n".join(output_lines) + "\n\n=== Training completed successfully ==="
                            else:
                                print(f"DEBUG: Training failed with return code: {return_code}")
                                final_output = "\n".join(output_lines) + f"\n\n=== Training failed with exit code {return_code} ==="
                            
                        except Exception as e:
                            print(f"DEBUG: Exception in training thread: {str(e)}")
                            final_output = f"Error during training: {str(e)}"
                        
                        finally:
                            self.is_training = False
                            self.training_process = None
                            print("DEBUG: Training thread finished")
                    
                    # Start training thread
                    self.training_thread = threading.Thread(target=run_training, daemon=True)
                    self.training_thread.start()
                    
                    return "Training started successfully", "Training started...\n\nCheck the terminal/console for detailed output."
                    
                except Exception as e:
                    self.is_training = False
                    error_msg = f"Error starting training: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg, error_msg
            
            def stop_training_function():
                """Stop the current training"""
                print("DEBUG: stop_training_function called")
                
                if not self.is_training:
                    return "No training is currently running", "Ready"
                
                try:
                    if self.training_process:
                        print(f"DEBUG: Terminating training process with PID: {self.training_process.pid}")
                        
                        # Try graceful termination first
                        self.training_process.terminate()
                        
                        # Wait a bit for graceful shutdown
                        try:
                            self.training_process.wait(timeout=10)
                            print("DEBUG: Training process terminated gracefully")
                        except subprocess.TimeoutExpired:
                            print("DEBUG: Graceful termination timed out, forcing kill")
                            self.training_process.kill()
                            self.training_process.wait()
                            print("DEBUG: Training process killed")
                    
                    self.is_training = False
                    self.training_process = None
                    
                    return "Training stopped", "Training stopped by user"
                    
                except Exception as e:
                    error_msg = f"Error stopping training: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    self.is_training = False
                    self.training_process = None
                    return error_msg, "Error occurred while stopping training"
            
            # Training execution event handlers
            execution_components["start_training_btn"].click(
                start_training_function,
                inputs=all_components,
                outputs=[execution_components["training_status"], execution_components["training_output"]]
            )
            print("DEBUG: Start training button event handler connected")
            
            execution_components["stop_training_btn"].click(
                stop_training_function,
                outputs=[execution_components["training_status"], execution_components["training_output"]]
            )
            print("DEBUG: Stop training button event handler connected")
            
            # Status refresh function
            def refresh_training_status():
                """Refresh training status"""
                status = self.get_training_status()
                return status
            
            execution_components["refresh_status_btn"].click(
                refresh_training_status,
                outputs=[execution_components["training_status"]]
            )
            print("DEBUG: Refresh status button event handler connected")
        
        return interface
    
    def _extract_full_config(self, *args):
        """Extract full configuration from component values"""
        print(f"DEBUG: _extract_full_config called with {len(args)} arguments")
        
        # Create a proper mapping of component values to config structure
        # The order should match the order in which components are added to all_components list
        try:
            # First, let's define the expected component order based on how we build all_components
            component_names = [
                # Model components (9 components)
                "task", "dit_path", "dit_high_noise_path", "t5_path", "clip_path", "vae_path",
                "timestep_boundary", "offload_inactive_dit", "lazy_loading",
                
                # Dataset components (19 components)
                "resolution_width", "resolution_height", "caption_extension", "dataset_batch_size",
                "enable_bucket", "bucket_no_upscale", "image_directory", "image_cache_directory",
                "image_num_repeats", "video_directory", "video_cache_directory", "target_frames",
                "frame_extraction", "frame_stride", "frame_sample", "max_frames", "source_fps",
                "video_num_repeats", "toml_output_path",
                
                # Training components (23 components)
                "learning_rate", "optimizer_type", "network_dim", "network_alpha",
                "max_train_epochs", "save_every_n_epochs", "batch_size", "seed",
                "timestep_sampling", "discrete_flow_shift", "min_timestep", "max_timestep",
                "preserve_distribution_shape", "mixed_precision", "blocks_to_swap",
                "fp8_base", "fp8_scaled", "fp8_t5", "gradient_checkpointing", "vae_cache_cpu",
                "max_data_loader_n_workers", "persistent_data_loader_workers", "attention_mode",
                "split_attn", "output_dir", "output_name", "dataset_config_path",
                "logging_dir", "log_with", "wandb_api_key", "wandb_run_name", "log_config",
                "debug_swapping",
                
                # Execution components (1 component)
                "text_encoder_batch_size"
            ]
            
            print(f"DEBUG: Expected {len(component_names)} components, got {len(args)} arguments")
            
            # Create a mapping of values
            values = {}
            for i, name in enumerate(component_names):
                if i < len(args):
                    values[name] = args[i]
                    print(f"DEBUG: {name} = {args[i]}")
                else:
                    values[name] = None
                    print(f"DEBUG: {name} = None (missing)")
            
            config = {
                "model": {
                    "task": values.get("task", "t2v-14B"),
                    "dit_path": values.get("dit_path", ""),
                    "dit_high_noise_path": values.get("dit_high_noise_path", ""),
                    "t5_path": values.get("t5_path", ""),
                    "clip_path": values.get("clip_path", ""),
                    "vae_path": values.get("vae_path", ""),
                    "timestep_boundary": values.get("timestep_boundary"),
                    "offload_inactive_dit": values.get("offload_inactive_dit", False),
                    "lazy_loading": values.get("lazy_loading", False)
                },
                "dataset": {
                    "resolution_width": values.get("resolution_width", 960),
                    "resolution_height": values.get("resolution_height", 544),
                    "caption_extension": values.get("caption_extension", ".txt"),
                    "dataset_batch_size": values.get("dataset_batch_size", 1),
                    "enable_bucket": values.get("enable_bucket", True),
                    "bucket_no_upscale": values.get("bucket_no_upscale", False),
                    "image_directory": values.get("image_directory", ""),
                    "image_cache_directory": values.get("image_cache_directory", ""),
                    "image_num_repeats": values.get("image_num_repeats", 1),
                    "video_directory": values.get("video_directory", ""),
                    "video_cache_directory": values.get("video_cache_directory", ""),
                    "target_frames": values.get("target_frames", 8),
                    "frame_extraction": values.get("frame_extraction", "uniform"),
                    "frame_stride": values.get("frame_stride", 1),
                    "frame_sample": values.get("frame_sample", "uniform"),
                    "max_frames": values.get("max_frames", 8),
                    "source_fps": values.get("source_fps", 24),
                    "video_num_repeats": values.get("video_num_repeats", 1),
                    "toml_output_path": values.get("toml_output_path", "dataset_config.toml")
                },
                "training": {
                    "learning_rate": values.get("learning_rate", 2e-4),
                    "optimizer_type": values.get("optimizer_type", "adamw8bit"),
                    "network_dim": values.get("network_dim", 32),
                    "network_alpha": values.get("network_alpha"),
                    "max_train_epochs": values.get("max_train_epochs", 16),
                    "save_every_n_epochs": values.get("save_every_n_epochs", 1),
                    "batch_size": values.get("batch_size", 1),
                    "seed": values.get("seed", 42),
                    "timestep_sampling": values.get("timestep_sampling", "shift"),
                    "discrete_flow_shift": values.get("discrete_flow_shift", 3.0),
                    "min_timestep": values.get("min_timestep"),
                    "max_timestep": values.get("max_timestep"),
                    "preserve_distribution_shape": values.get("preserve_distribution_shape", False),
                    "mixed_precision": values.get("mixed_precision", "bf16"),
                    "blocks_to_swap": values.get("blocks_to_swap"),
                    "fp8_base": values.get("fp8_base", True),
                    "fp8_scaled": values.get("fp8_scaled", False),
                    "fp8_t5": values.get("fp8_t5", False),
                    "gradient_checkpointing": values.get("gradient_checkpointing", True),
                    "vae_cache_cpu": values.get("vae_cache_cpu", False),
                    "max_data_loader_n_workers": values.get("max_data_loader_n_workers", 2),
                    "persistent_data_loader_workers": values.get("persistent_data_loader_workers", True),
                    "attention_mode": values.get("attention_mode", "sdpa"),
                    "split_attn": values.get("split_attn", False),
                    "output_dir": values.get("output_dir", "outputs"),
                    "output_name": values.get("output_name", "wan_lora"),
                    "dataset_config_path": values.get("dataset_config_path", ""),
                    "logging_dir": values.get("logging_dir", ""),
                    "log_with": values.get("log_with", ""),
                    "wandb_api_key": values.get("wandb_api_key", ""),
                    "wandb_run_name": values.get("wandb_run_name", ""),
                    "log_config": values.get("log_config", False),
                    "debug_swapping": values.get("debug_swapping", False)
                },
                "execution": {
                    "text_encoder_batch_size": values.get("text_encoder_batch_size", 16)
                }
            }
            
            print(f"DEBUG: Built config successfully")
            return config
            
        except Exception as e:
            print(f"DEBUG: Error in _extract_full_config: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a basic fallback config
            return {
                "model": {
                    "task": "t2v-14B",
                    "dit_path": "",
                    "dit_high_noise_path": "",
                    "t5_path": "",
                    "clip_path": "",
                    "vae_path": "",
                    "timestep_boundary": None,
                    "offload_inactive_dit": False,
                    "lazy_loading": False
                },
                "training": {
                    "learning_rate": 2e-4,
                    "optimizer_type": "adamw8bit",
                    "network_dim": 32,
                    "network_alpha": None,
                    "max_train_epochs": 16,
                    "save_every_n_epochs": 1,
                    "batch_size": 1,
                    "seed": 42,
                    "timestep_sampling": "shift",
                    "discrete_flow_shift": 3.0,
                    "min_timestep": None,
                    "max_timestep": None,
                    "preserve_distribution_shape": False,
                    "mixed_precision": "bf16",
                    "blocks_to_swap": None,
                    "fp8_base": True,
                    "fp8_scaled": False,
                    "fp8_t5": False,
                    "gradient_checkpointing": True,
                    "vae_cache_cpu": False,
                    "max_data_loader_n_workers": 2,
                    "persistent_data_loader_workers": True,
                    "attention_mode": "sdpa",
                    "split_attn": False,
                    "output_dir": "outputs",
                    "output_name": "wan_lora",
                    "dataset_config_path": "",
                    "logging_dir": "",
                    "log_with": "",
                    "wandb_api_key": "",
                    "wandb_run_name": "",
                    "log_config": False
                },
                "execution": {
                    "vae_chunk_size": 32,
                    "vae_tiling": True,
                    "text_encoder_batch_size": 16
                }
            }
    
    def _config_to_component_updates(self, config):
        """Convert config back to component updates"""
        print(f"DEBUG: _config_to_component_updates called")
        
        try:
            model_config = config.get("model", {})
            dataset_config = config.get("dataset", {})
            training_config = config.get("training", {})
            execution_config = config.get("execution", {})
            
            # Create updates in the same order as component_names in _extract_full_config
            updates = []
            
            # Model components (9 components)
            updates.append(gr.update(value=model_config.get("task", "t2v-14B")))
            updates.append(gr.update(value=model_config.get("dit_path", "")))
            updates.append(gr.update(value=model_config.get("dit_high_noise_path", "")))
            updates.append(gr.update(value=model_config.get("t5_path", "")))
            updates.append(gr.update(value=model_config.get("clip_path", "")))
            updates.append(gr.update(value=model_config.get("vae_path", "")))
            updates.append(gr.update(value=model_config.get("timestep_boundary")))
            updates.append(gr.update(value=model_config.get("offload_inactive_dit", False)))
            updates.append(gr.update(value=model_config.get("lazy_loading", False)))
            
            # Dataset components (19 components)
            updates.append(gr.update(value=dataset_config.get("resolution_width", 960)))
            updates.append(gr.update(value=dataset_config.get("resolution_height", 544)))
            updates.append(gr.update(value=dataset_config.get("caption_extension", ".txt")))
            updates.append(gr.update(value=dataset_config.get("dataset_batch_size", 1)))
            updates.append(gr.update(value=dataset_config.get("enable_bucket", True)))
            updates.append(gr.update(value=dataset_config.get("bucket_no_upscale", False)))
            updates.append(gr.update(value=dataset_config.get("image_directory", "")))
            updates.append(gr.update(value=dataset_config.get("image_cache_directory", "")))
            updates.append(gr.update(value=dataset_config.get("image_num_repeats", 1)))
            updates.append(gr.update(value=dataset_config.get("video_directory", "")))
            updates.append(gr.update(value=dataset_config.get("video_cache_directory", "")))
            updates.append(gr.update(value=dataset_config.get("target_frames", 8)))
            updates.append(gr.update(value=dataset_config.get("frame_extraction", "uniform")))
            updates.append(gr.update(value=dataset_config.get("frame_stride", 1)))
            updates.append(gr.update(value=dataset_config.get("frame_sample", "uniform")))
            updates.append(gr.update(value=dataset_config.get("max_frames", 8)))
            updates.append(gr.update(value=dataset_config.get("source_fps", 24)))
            updates.append(gr.update(value=dataset_config.get("video_num_repeats", 1)))
            updates.append(gr.update(value=dataset_config.get("toml_output_path", "dataset_config.toml")))
            
            # Training components (23 components)
            updates.append(gr.update(value=training_config.get("learning_rate", 2e-4)))
            updates.append(gr.update(value=training_config.get("optimizer_type", "adamw8bit")))
            updates.append(gr.update(value=training_config.get("network_dim", 32)))
            updates.append(gr.update(value=training_config.get("network_alpha")))
            updates.append(gr.update(value=training_config.get("max_train_epochs", 16)))
            updates.append(gr.update(value=training_config.get("save_every_n_epochs", 1)))
            updates.append(gr.update(value=training_config.get("batch_size", 1)))
            updates.append(gr.update(value=training_config.get("seed", 42)))
            updates.append(gr.update(value=training_config.get("timestep_sampling", "shift")))
            updates.append(gr.update(value=training_config.get("discrete_flow_shift", 3.0)))
            updates.append(gr.update(value=training_config.get("min_timestep")))
            updates.append(gr.update(value=training_config.get("max_timestep")))
            updates.append(gr.update(value=training_config.get("preserve_distribution_shape", False)))
            updates.append(gr.update(value=training_config.get("mixed_precision", "bf16")))
            updates.append(gr.update(value=training_config.get("blocks_to_swap")))
            updates.append(gr.update(value=training_config.get("fp8_base", True)))
            updates.append(gr.update(value=training_config.get("fp8_scaled", False)))
            updates.append(gr.update(value=training_config.get("fp8_t5", False)))
            updates.append(gr.update(value=training_config.get("gradient_checkpointing", True)))
            updates.append(gr.update(value=training_config.get("vae_cache_cpu", False)))
            updates.append(gr.update(value=training_config.get("max_data_loader_n_workers", 2)))
            updates.append(gr.update(value=training_config.get("persistent_data_loader_workers", True)))
            updates.append(gr.update(value=training_config.get("attention_mode", "sdpa")))
            updates.append(gr.update(value=training_config.get("split_attn", False)))
            updates.append(gr.update(value=training_config.get("output_dir", "outputs")))
            updates.append(gr.update(value=training_config.get("output_name", "wan_lora")))
            updates.append(gr.update(value=training_config.get("dataset_config_path", "")))
            updates.append(gr.update(value=training_config.get("logging_dir", "")))
            updates.append(gr.update(value=training_config.get("log_with", "")))
            updates.append(gr.update(value=training_config.get("wandb_api_key", "")))
            updates.append(gr.update(value=training_config.get("wandb_run_name", "")))
            updates.append(gr.update(value=training_config.get("log_config", False)))
            updates.append(gr.update(value=training_config.get("debug_swapping", False)))
            
            # Execution components (1 component)
            updates.append(gr.update(value=execution_config.get("text_encoder_batch_size", 16)))
            
            print(f"DEBUG: Generated {len(updates)} component updates (expected 52: 9 model + 19 dataset + 23 training + 1 execution)")
            return updates
            
        except Exception as e:
            print(f"DEBUG: Error in _config_to_component_updates: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return correct number of empty updates as fallback
            return [gr.update() for _ in range(53)]

    def load_caption_model(self, model_name: str) -> str:
        """Load caption generation model"""
        if not CAPTION_AVAILABLE:
            return "Error: transformers library not available. Please install: pip install transformers torch pillow"
        
        try:
            if model_name == "BLIP-2":
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                self.caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            elif model_name == "BLIP":
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            elif model_name == "Git-Large":
                self.caption_pipeline = pipeline("image-to-text", model="microsoft/git-large-coco")
            else:
                return "Error: Unknown model selected"
            
            return f"Successfully loaded {model_name} model"
            
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def generate_caption_for_image(self, image_path: str, model_name: str, custom_prompt: str = "") -> str:
        """Generate caption for a single image"""
        if not CAPTION_AVAILABLE:
            return "Error: transformers library not available"
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if model_name == "Git-Large" and self.caption_pipeline:
                result = self.caption_pipeline(image)
                caption = result[0]['generated_text']
            elif self.caption_model and self.caption_processor:
                if custom_prompt:
                    inputs = self.caption_processor(image, custom_prompt, return_tensors="pt")
                else:
                    inputs = self.caption_processor(image, return_tensors="pt")
                
                with torch.no_grad():
                    generated_ids = self.caption_model.generate(**inputs, max_length=50)
                    caption = self.caption_processor.decode(generated_ids[0], skip_special_tokens=True)
            else:
                return "Error: No model loaded"
            
            return caption.strip()
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"

    def batch_generate_captions(self, folder_path: str, model_name: str, custom_prompt: str = "", 
                              file_extensions: str = "jpg,jpeg,png,webp") -> str:
        """Generate captions for all images in a folder"""
        if not os.path.isdir(folder_path):
            return "Error: Invalid folder path"
        
        extensions = [ext.strip().lower() for ext in file_extensions.split(',')]
        image_files = []
        
        # Find all image files
        for ext in extensions:
            image_files.extend(Path(folder_path).glob(f"*.{ext}"))
            image_files.extend(Path(folder_path).glob(f"*.{ext.upper()}"))
        
        if not image_files:
            return f"No image files found with extensions: {file_extensions}"
        
        processed = 0
        errors = 0
        
        for image_file in image_files:
            try:
                caption = self.generate_caption_for_image(str(image_file), model_name, custom_prompt)
                
                if not caption.startswith("Error"):
                    # Save caption to .txt file
                    caption_file = image_file.with_suffix('.txt')
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    processed += 1
                else:
                    errors += 1
                    print(f"Error processing {image_file}: {caption}")
                    
            except Exception as e:
                errors += 1
                print(f"Error processing {image_file}: {str(e)}")
        
        return f"Processed {processed} images successfully. {errors} errors occurred."

    def create_caption_interface(self):
        """Create the caption generation interface"""
        with gr.TabItem("Caption Generator"):
            with gr.Column():
                gr.Markdown("## 🏷️ Caption Generator for Training Datasets")
                gr.Markdown("Generate captions for your training images using pre-trained vision-language models.")
                
                if not CAPTION_AVAILABLE:
                    gr.Markdown("⚠️ **Missing Dependencies**: Please install required packages:")
                    gr.Code("pip install transformers torch pillow", language="bash")
                
                with gr.Row():
                    with gr.Column():
                        model_choice = gr.Dropdown(
                            choices=["BLIP", "BLIP-2", "Git-Large"],
                            value="BLIP",
                            label="Caption Model",
                            info="Choose the model for caption generation"
                        )
                        
                        load_model_btn = gr.Button("Load Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                    
                    with gr.Column():
                        custom_prompt = gr.Textbox(
                            label="Custom Prompt (Optional)",
                            placeholder="e.g., 'a photo of' or 'describe this anime character'",
                            info="Custom prompt to guide caption generation"
                        )
                
                with gr.Tabs():
                    with gr.TabItem("Single Image"):
                        with gr.Row():
                            with gr.Column():
                                single_image = gr.Image(
                                    label="Upload Image",
                                    type="filepath"
                                )
                                single_caption_btn = gr.Button("Generate Caption")
                            
                            with gr.Column():
                                single_caption_output = gr.Textbox(
                                    label="Generated Caption",
                                    lines=3
                                )
                    
                    with gr.TabItem("Batch Processing"):
                        with gr.Column():
                            batch_folder = gr.Textbox(
                                label="Image Folder Path",
                                placeholder="C:/path/to/your/images",
                                info="Folder containing images to caption"
                            )
                            
                            file_extensions = gr.Textbox(
                                label="File Extensions",
                                value="jpg,jpeg,png,webp",
                                info="Comma-separated list of file extensions"
                            )
                            
                            batch_caption_btn = gr.Button("Generate All Captions", variant="primary")
                            batch_status = gr.Textbox(
                                label="Batch Status",
                                lines=3,
                                interactive=False
                            )
                
                with gr.Accordion("Training Prompt Examples", open=False):
                    gr.Markdown("""
                    ### Image Training Examples:
                    **Character Training:**
                    ```
                    1girl, solo, long_hair, blue_eyes, blonde_hair, school_uniform, white_shirt, blue_skirt, sitting, classroom, looking_at_viewer, smile
                    ```
                    
                    **Style Training:**
                    ```
                    anime_style, cel_shading, vibrant_colors, clean_lineart, 1girl, school_uniform, cherry_blossoms
                    ```
                    
                    **Realistic Portrait:**
                    ```
                    portrait, 1boy, mature_male, facial_hair, serious_expression, detailed_eyes, realistic, studio_lighting
                    ```
                    
                    ### Video Training Examples:
                    **Movement:**
                    ```
                    1girl, walking, full_body, side_view, natural_gait, outdoor, path, trees, smooth_motion
                    ```
                    
                    **Scene:**
                    ```
                    forest, trees_swaying, wind_effect, natural_lighting, peaceful_atmosphere, birds_flying
                    ```
                    
                    ### Tips:
                    - For character training: Remove character name and distinctive features from captions
                    - For style training: Remove artist names but keep style descriptors
                    - Use consistent tagging format across your dataset
                    - Add activation tags for specific concepts
                    """)
        
        # Event handlers
        load_model_btn.click(
            fn=self.load_caption_model,
            inputs=[model_choice],
            outputs=[model_status]
        )
        
        single_caption_btn.click(
            fn=self.generate_caption_for_image,
            inputs=[single_image, model_choice, custom_prompt],
            outputs=[single_caption_output]
        )
        
        batch_caption_btn.click(
            fn=self.batch_generate_captions,
            inputs=[batch_folder, model_choice, custom_prompt, file_extensions],
            outputs=[batch_status]
        )

def main():
    """Main function to launch the UI"""
    app = MusubiTrainerUI()
    interface = app.create_interface()
    
    try:
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()
