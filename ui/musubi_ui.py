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
- Image/Video inference with LoRA support
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
        
        # Inference process state
        self.inference_process = None
        self.inference_thread = None
        self.is_inference_running = False
        
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
        
        if self.is_inference_running and self.inference_process:
            print("DEBUG: Cleaning up inference process...")
            try:
                self.inference_process.terminate()
                self.inference_process.wait(timeout=5)
            except:
                try:
                    self.inference_process.kill()
                except:
                    pass
            self.is_inference_running = False
            self.inference_process = None
        
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
            
            # Show current config directory
            gr.Markdown(f"**Configuration Directory**: `{self.config_manager.config_dir}`")
            gr.Markdown("*You can change this by setting `CONFIG_DIR` in `.env`*")
            
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
    
    def create_inference_ui(self):
        """Create comprehensive inference UI for I2V and video generation with WAN 2.1/2.2 support"""
        with gr.Group():
            gr.Markdown("## Inference Configuration")
            gr.Markdown("Generate images/videos using trained LoRA models with WAN 2.1/2.2")
            
            with gr.Tabs():
                # I2V One Frame Inference Tab
                with gr.TabItem("I2V (1-Frame Inference)"):
                    gr.Markdown("""
                    ### Image-to-Video Single Frame Inference
                    
                    **Note**: 1-frame inference requires a trained LoRA model. This feature is experimental.
                    
                    **Usage**: Input a starting image and prompt to generate a single transformed frame.
                    The generated image will show temporal/semantic changes based on your prompt.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            # Model Configuration
                            gr.Markdown("#### Model Configuration")
                            
                            i2v_wan_version = gr.Dropdown(
                                choices=[("WAN 2.1", "2.1"), ("WAN 2.2", "2.2")],
                                value="2.1",
                                label="WAN Version",
                                info="Select WAN model version"
                            )
                            
                            i2v_task = gr.Dropdown(
                                choices=[("I2V 14B", "i2v-14B"), ("I2V A14B (WAN 2.2)", "i2v-A14B")],
                                value="i2v-14B",
                                label="Task",
                                info="Use I2V 14B model for 1-frame inference"
                            )
                            
                            i2v_dit_path = gr.Textbox(
                                label="DiT Model Path *",
                                placeholder="path/to/wan2.x_i2v_model.safetensors",
                                info="I2V DiT model weights (required)"
                            )
                            
                            i2v_dit_high_noise_path = gr.Textbox(
                                label="DiT High Noise Path (WAN 2.2)",
                                placeholder="path/to/wan2.2_i2v_high_noise_model.safetensors",
                                info="High noise DiT model for WAN 2.2",
                                visible=False
                            )
                            
                            i2v_vae_path = gr.Textbox(
                                label="VAE Model Path *", 
                                placeholder="path/to/wan_2.1_vae.safetensors",
                                info="VAE model for encoding/decoding (required)"
                            )
                            
                            i2v_t5_path = gr.Textbox(
                                label="T5 Text Encoder Path *",
                                placeholder="path/to/models_t5_umt5-xxl-enc-bf16.pth",
                                info="T5 text encoder model (required)"
                            )
                            
                            i2v_clip_path = gr.Textbox(
                                label="CLIP Model Path (WAN 2.1 only)",
                                placeholder="path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                                info="CLIP model required for WAN 2.1"
                            )
                            
                            i2v_lora_path = gr.Textbox(
                                label="LoRA Model Path",
                                placeholder="path/to/lora_model.safetensors",
                                info="Optional but recommended: Trained LoRA model for better 1-frame inference quality"
                            )
                            
                            i2v_lora_multiplier = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="LoRA Multiplier",
                                info="Strength of LoRA effects"
                            )
                        
                        with gr.Column():
                            # Inference Configuration
                            gr.Markdown("#### Inference Settings")
                            i2v_input_image = gr.Image(
                                label="Input Image *",
                                type="filepath",
                                # info="Starting image for transformation (required)"
                            )
                            
                            i2v_prompt = gr.Textbox(
                                label="Prompt *",
                                placeholder="A cat wearing sunglasses",
                                lines=3,
                                info="Describe the desired transformation (required)"
                            )
                            
                            i2v_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="blurry, low quality, distorted",
                                lines=2,
                                value="blurry, low quality, distorted",
                                info="What to avoid in generation"
                            )
                            
                            with gr.Row():
                                i2v_width = gr.Number(
                                    value=384,
                                    label="Width",
                                    info="Output image width"
                                )
                                i2v_height = gr.Number(
                                    value=576,
                                    label="Height", 
                                    info="Output image height"
                                )
                            
                            with gr.Row():
                                i2v_steps = gr.Slider(
                                    minimum=10,
                                    maximum=50,
                                    value=25,
                                    step=1,
                                    label="Inference Steps",
                                    info="Number of denoising steps"
                                )
                                i2v_guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.0,
                                    step=0.5,
                                    label="Guidance Scale",
                                    info="Adherence to prompt"
                                )
                            
                            with gr.Row():
                                i2v_target_index = gr.Number(
                                    value=1,
                                    label="Target Index",
                                    info="RoPE timestamp for generated frame (>=1)"
                                )
                                i2v_control_index = gr.Number(
                                    value=0,
                                    label="Control Index", 
                                    info="RoPE timestamp for control frame (usually 0)"
                                )
                            
                            i2v_seed = gr.Number(
                                value=-1,
                                label="Seed",
                                info="Random seed (-1 for random)"
                            )
                
                # Standard Video Generation Tab  
                with gr.TabItem("Video Generation"):
                    gr.Markdown("""
                    ### Standard Video Generation
                    
                    Generate videos using Text-to-Video or Image-to-Video models.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            # Model Configuration
                            gr.Markdown("#### Model Configuration")
                            
                            video_wan_version = gr.Dropdown(
                                choices=[("WAN 2.1", "2.1"), ("WAN 2.2", "2.2")],
                                value="2.1",
                                label="WAN Version",
                                info="Select WAN model version"
                            )
                            
                            video_task = gr.Dropdown(
                                choices=[
                                    ("Text-to-Video 14B", "t2v-14B"),
                                    ("Image-to-Video 14B", "i2v-14B"),
                                    ("FLF2V 14B", "flf2v-14B"),
                                    ("Text-to-Video A14B (WAN 2.2)", "t2v-A14B"),
                                    ("Image-to-Video A14B (WAN 2.2)", "i2v-A14B"),
                                    ("FLF2V A14B (WAN 2.2)", "flf2v-A14B")
                                ],
                                value="t2v-14B",
                                label="Task",
                                info="Select video generation task"
                            )
                            
                            video_dit_path = gr.Textbox(
                                label="DiT Model Path *",
                                placeholder="path/to/wan2.x_model.safetensors",
                                info="Main DiT model weights (required)"
                            )
                            
                            video_dit_high_noise_path = gr.Textbox(
                                label="DiT High Noise Path (WAN 2.2)",
                                placeholder="path/to/wan2.2_high_noise_model.safetensors",
                                info="High noise DiT model for WAN 2.2",
                                visible=False
                            )
                            
                            video_vae_path = gr.Textbox(
                                label="VAE Model Path *",
                                placeholder="path/to/wan_2.1_vae.safetensors",
                                info="VAE model (required)"
                            )
                            
                            video_t5_path = gr.Textbox(
                                label="T5 Text Encoder Path *",
                                placeholder="path/to/models_t5_umt5-xxl-enc-bf16.pth",
                                info="T5 text encoder (required)"
                            )
                            
                            video_clip_path = gr.Textbox(
                                label="CLIP Model Path (WAN 2.1 only)",
                                placeholder="path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                                info="CLIP model required for WAN 2.1"
                            )
                            
                            video_lora_path = gr.Textbox(
                                label="LoRA Model Path (Optional)",
                                placeholder="path/to/lora_model.safetensors",
                                info="Optional: Apply trained LoRA"
                            )
                            
                            video_lora_multiplier = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="LoRA Multiplier"
                            )
                        
                        with gr.Column():
                            # Generation Settings
                            gr.Markdown("#### Generation Settings")
                            video_input_image = gr.Image(
                                label="Input Image (I2V only)",
                                type="filepath",
                                # info="Starting image for I2V generation"
                            )
                            
                            video_prompt = gr.Textbox(
                                label="Prompt *",
                                placeholder="A beautiful landscape with flowing water",
                                lines=3,
                                info="Generation prompt (required)"
                            )
                            
                            video_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="blurry, low quality, distorted",
                                lines=2,
                                value="blurry, low quality, distorted"
                            )
                            
                            with gr.Row():
                                video_width = gr.Number(
                                    value=256,
                                    label="Width"
                                )
                                video_height = gr.Number(
                                    value=256,
                                    label="Height"
                                )
                                video_length = gr.Number(
                                    value=16,
                                    label="Video Length (frames)"
                                )
                            
                            with gr.Row():
                                video_steps = gr.Slider(
                                    minimum=10,
                                    maximum=50,
                                    value=25,
                                    step=1,
                                    label="Inference Steps"
                                )
                                video_guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.0,
                                    step=0.5,
                                    label="Guidance Scale"
                                )
                            
                            video_seed = gr.Number(
                                value=-1,
                                label="Seed",
                                info="Random seed (-1 for random)"
                            )
            
            # Advanced Settings (shared for both tabs)
            with gr.Group():
                gr.Markdown("#### Advanced Settings")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Performance Options**")
                        
                        attn_mode = gr.Dropdown(
                            choices=[
                                ("PyTorch SDPA (Default)", "torch"),
                                ("Flash Attention 2", "flash2"),
                                ("Flash Attention 3", "flash3"),
                                ("xFormers", "xformers"),
                                ("Sage Attention", "sageattn")
                            ],
                            value="torch",
                            label="Attention Mode",
                            info="Attention mechanism to use"
                        )
                        
                        blocks_to_swap = gr.Slider(
                            minimum=0,
                            maximum=39,
                            value=0,
                            step=1,
                            label="Blocks to Swap",
                            info="Number of DiT blocks to swap to CPU (0=disable)"
                        )
                        
                        enable_compile = gr.Checkbox(
                            label="Enable torch.compile",
                            value=False,
                            info="Enable torch compilation for speed (experimental)"
                        )
                    
                    with gr.Column():
                        gr.Markdown("**FP8 Options**")
                        
                        fp8_base = gr.Checkbox(
                            label="FP8 Base",
                            value=False,
                            info="Run DiT in FP8 mode (saves VRAM)"
                        )
                        
                        fp8_scaled = gr.Checkbox(
                            label="FP8 Scaled",
                            value=False,
                            info="FP8 weight optimization (requires FP8 Base)"
                        )
                        
                        fp8_fast = gr.Checkbox(
                            label="FP8 Fast (RTX 40x0)",
                            value=False,
                            info="Fastest FP8 mode for RTX 40x0 (may reduce quality)"
                        )
                        
                        fp8_t5 = gr.Checkbox(
                            label="FP8 T5",
                            value=False,
                            info="Run T5 text encoder in FP8 mode"
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**WAN 2.2 Options**")
                        
                        timestep_boundary = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.875,
                            step=0.01,
                            label="Timestep Boundary",
                            info="Boundary for switching between models (WAN 2.2)"
                        )
                        
                        guidance_scale_high_noise = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.0,
                            step=0.5,
                            label="Guidance Scale (High Noise)",
                            info="Separate guidance scale for high noise model"
                        )
                        
                        offload_inactive_dit = gr.Checkbox(
                            label="Offload Inactive DiT",
                            value=False,
                            info="Move inactive DiT to CPU (saves VRAM)"
                        )
                        
                        lazy_loading = gr.Checkbox(
                            label="Lazy Loading",
                            value=False,
                            info="Enable lazy loading for DiT models"
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Memory Options**")
                        
                        vae_cache_cpu = gr.Checkbox(
                            label="VAE Cache CPU",
                            value=False,
                            info="Cache VAE features in CPU memory"
                        )
                        
                        flow_shift = gr.Number(
                            value=3.0,
                            label="Flow Shift",
                            info="Flow shift parameter (3.0 for I2V 480p, 5.0 for others)"
                        )
                        
                        trim_tail_frames = gr.Number(
                            value=0,
                            label="Trim Tail Frames",
                            info="Number of frames to trim from end"
                        )
            
            # Common inference controls
            with gr.Group():
                gr.Markdown("#### Execution")
                with gr.Row():
                    inference_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="outputs/inference",
                        info="Directory to save generated content"
                    )
                
                with gr.Row():
                    output_type = gr.Dropdown(
                        label="Output Type",
                        choices=["images", "latent", "both", "latent_images"],
                        value="latent_images",
                        info="Output format: images only, latents only, both, or latents+images (best for debugging)"
                    )
                
                # VAE Decode Section
                with gr.Group():
                    gr.Markdown("#### VAE Decode & Debug")
                    with gr.Row():
                        latent_file_path = gr.Textbox(
                            label="Latent File Path",
                            placeholder="outputs/inference/timestamp_seed_latent.safetensors",
                            info="Path to latent file for debugging VAE decode"
                        )
                        generate_vae_decode_btn = gr.Button("Generate VAE Decode Command", variant="secondary")
                    
                    with gr.Row():
                        inspect_latents_btn = gr.Button("Inspect Latent Values", variant="secondary")
                        debug_vae_stats_btn = gr.Button("VAE Stats & Preprocessing", variant="secondary")
                    
                    vae_decode_command_output = gr.Textbox(
                        label="VAE Decode Command",
                        lines=3,
                        interactive=False,
                        info="Command to decode latents for debugging"
                    )
                    
                    latent_debug_output = gr.Textbox(
                        label="Latent Debug Info",
                        lines=5,
                        interactive=False,
                        info="Latent statistics and VAE preprocessing debug info"
                    )
                
                with gr.Row():
                    generate_i2v_btn = gr.Button("Generate I2V (1-Frame)", variant="primary", size="lg")
                    generate_video_btn = gr.Button("Generate Video", variant="primary", size="lg")
                    stop_inference_btn = gr.Button("Stop Generation", variant="stop")
                
                inference_command_output = gr.Textbox(
                    label="Generated Command",
                    lines=5,
                    interactive=False,
                    info="Command that will be executed"
                )
                
                inference_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                inference_output = gr.Textbox(
                    label="Generation Output",
                    lines=10,
                    interactive=False,
                    info="Real-time output from generation process"
                )
            
            # Configuration management
            with gr.Group():
                gr.Markdown("#### Configuration Management")
                with gr.Row():
                    inference_config_name = gr.Textbox(
                        label="Config Name",
                        placeholder="Enter configuration name",
                        info="Name for saving/loading configurations"
                    )
                    inference_config_list = gr.Dropdown(
                        label="Saved Configurations",
                        choices=[],
                        interactive=True,
                        info="Select a saved configuration to load"
                    )
                
                with gr.Row():
                    save_inference_config_btn = gr.Button("Save Config", variant="secondary", size="sm")
                    load_inference_config_btn = gr.Button("Load Config", variant="secondary", size="sm")
                    delete_inference_config_btn = gr.Button("Delete Config", variant="stop", size="sm")
                    refresh_inference_configs_btn = gr.Button("Refresh", variant="secondary", size="sm")
                
                inference_config_status = gr.Textbox(
                    label="Config Status",
                    value="Ready",
                    interactive=False,
                    lines=1
                )
        
        # Add event handlers for WAN version changes
        def update_i2v_ui_for_version(wan_version):
            """Update I2V UI based on WAN version"""
            is_wan22 = wan_version == "2.2"
            # Keep all choices available for configuration loading, but set appropriate default
            all_choices = [("I2V 14B", "i2v-14B"), ("I2V A14B (WAN 2.2)", "i2v-A14B")]
            default_value = "i2v-A14B" if is_wan22 else "i2v-14B"
            
            return {
                i2v_task: gr.Dropdown(choices=all_choices, value=default_value),
                i2v_dit_high_noise_path: gr.Textbox(visible=is_wan22),
                i2v_clip_path: gr.Textbox(visible=not is_wan22)
            }
        
        def update_video_ui_for_version(wan_version):
            """Update video UI based on WAN version"""
            is_wan22 = wan_version == "2.2"
            # Keep all choices available for configuration loading
            all_choices = [
                ("Text-to-Video 14B", "t2v-14B"),
                ("Image-to-Video 14B", "i2v-14B"),
                ("FLF2V 14B", "flf2v-14B"),
                ("Text-to-Video A14B (WAN 2.2)", "t2v-A14B"),
                ("Image-to-Video A14B (WAN 2.2)", "i2v-A14B"),
                ("FLF2V A14B (WAN 2.2)", "flf2v-A14B")
            ]
            default_value = "t2v-A14B" if is_wan22 else "t2v-14B"
            
            return {
                video_task: gr.Dropdown(choices=all_choices, value=default_value),
                video_dit_high_noise_path: gr.Textbox(visible=is_wan22),
                video_clip_path: gr.Textbox(visible=not is_wan22)
            }
        
        def update_task_defaults(task):
            """Update task-specific defaults for timestep boundary, guidance scale, and flow shift"""
            task_defaults = {
                # WAN 2.1 tasks
                "t2v-14B": {"timestep_boundary": 0.875, "guidance_scale": 7.0, "guidance_scale_high_noise": 7.0, "flow_shift": 12.0},
                "i2v-14B": {"timestep_boundary": 0.900, "guidance_scale": 3.5, "guidance_scale_high_noise": 3.5, "flow_shift": 5.0},
                "flf2v-14B": {"timestep_boundary": 0.900, "guidance_scale": 3.5, "guidance_scale_high_noise": 3.5, "flow_shift": 5.0},
                # WAN 2.2 tasks  
                "t2v-A14B": {"timestep_boundary": 0.875, "guidance_scale": 3.0, "guidance_scale_high_noise": 4.0, "flow_shift": 12.0},
                "i2v-A14B": {"timestep_boundary": 0.900, "guidance_scale": 3.5, "guidance_scale_high_noise": 3.5, "flow_shift": 5.0},
                "flf2v-A14B": {"timestep_boundary": 0.900, "guidance_scale": 3.5, "guidance_scale_high_noise": 3.5, "flow_shift": 5.0},
            }
            
            defaults = task_defaults.get(task, task_defaults["t2v-14B"])
            return (
                defaults["timestep_boundary"],
                defaults["guidance_scale"], 
                defaults["guidance_scale_high_noise"],
                defaults["flow_shift"]
            )
        
        i2v_wan_version.change(
            update_i2v_ui_for_version,
            inputs=[i2v_wan_version],
            outputs=[i2v_task, i2v_dit_high_noise_path, i2v_clip_path]
        )
        
        video_wan_version.change(
            update_video_ui_for_version,
            inputs=[video_wan_version],
            outputs=[video_task, video_dit_high_noise_path, video_clip_path]
        )
        
        # Add task change handlers to update defaults
        video_task.change(
            update_task_defaults,
            inputs=[video_task],
            outputs=[timestep_boundary, video_guidance_scale, guidance_scale_high_noise, flow_shift]
        )
        
        return {
            # I2V components
            "i2v_wan_version": i2v_wan_version,
            "i2v_task": i2v_task,
            "i2v_dit_path": i2v_dit_path,
            "i2v_dit_high_noise_path": i2v_dit_high_noise_path,
            "i2v_vae_path": i2v_vae_path,
            "i2v_t5_path": i2v_t5_path,
            "i2v_clip_path": i2v_clip_path,
            "i2v_lora_path": i2v_lora_path,
            "i2v_lora_multiplier": i2v_lora_multiplier,
            "i2v_input_image": i2v_input_image,
            "i2v_prompt": i2v_prompt,
            "i2v_negative_prompt": i2v_negative_prompt,
            "i2v_width": i2v_width,
            "i2v_height": i2v_height,
            "i2v_steps": i2v_steps,
            "i2v_guidance_scale": i2v_guidance_scale,
            "i2v_target_index": i2v_target_index,
            "i2v_control_index": i2v_control_index,
            "i2v_seed": i2v_seed,
            # Video components
            "video_wan_version": video_wan_version,
            "video_task": video_task,
            "video_dit_path": video_dit_path,
            "video_dit_high_noise_path": video_dit_high_noise_path,
            "video_vae_path": video_vae_path,
            "video_t5_path": video_t5_path,
            "video_clip_path": video_clip_path,
            "video_lora_path": video_lora_path,
            "video_lora_multiplier": video_lora_multiplier,
            "video_input_image": video_input_image,
            "video_prompt": video_prompt,
            "video_negative_prompt": video_negative_prompt,
            "video_width": video_width,
            "video_height": video_height,
            "video_length": video_length,
            "video_steps": video_steps,
            "video_guidance_scale": video_guidance_scale,
            "video_seed": video_seed,
            # Advanced options
            "attn_mode": attn_mode,
            "blocks_to_swap": blocks_to_swap,
            "enable_compile": enable_compile,
            "fp8_base": fp8_base,
            "fp8_scaled": fp8_scaled,
            "fp8_fast": fp8_fast,
            "fp8_t5": fp8_t5,
            "timestep_boundary": timestep_boundary,
            "guidance_scale_high_noise": guidance_scale_high_noise,
            "offload_inactive_dit": offload_inactive_dit,
            "lazy_loading": lazy_loading,
            "vae_cache_cpu": vae_cache_cpu,
            "flow_shift": flow_shift,
            "trim_tail_frames": trim_tail_frames,
            # Common components
            "inference_output_dir": inference_output_dir,
            "output_type": output_type,
            "latent_file_path": latent_file_path,
            "generate_vae_decode_btn": generate_vae_decode_btn,
            "inspect_latents_btn": inspect_latents_btn,
            "debug_vae_stats_btn": debug_vae_stats_btn,
            "vae_decode_command_output": vae_decode_command_output,
            "latent_debug_output": latent_debug_output,
            "generate_i2v_btn": generate_i2v_btn,
            "generate_video_btn": generate_video_btn,
            "stop_inference_btn": stop_inference_btn,
            "inference_command_output": inference_command_output,
            "inference_status": inference_status,
            "inference_output": inference_output,
            # Configuration management components
            "inference_config_name": inference_config_name,
            "inference_config_list": inference_config_list,
            "save_inference_config_btn": save_inference_config_btn,
            "load_inference_config_btn": load_inference_config_btn,
            "delete_inference_config_btn": delete_inference_config_btn,
            "refresh_inference_configs_btn": refresh_inference_configs_btn,
            "inference_config_status": inference_config_status
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
    
    def build_i2v_inference_command(self, inference_config):
        """Build command for I2V 1-frame inference with comprehensive WAN support"""
        try:
            venv_python = ".venv\\Scripts\\python.exe"
            cmd = [venv_python, "src/musubi_tuner/wan_generate_video.py"]
            
            # Task (always use I2V for 1-frame inference)
            task = inference_config.get("i2v_task", "i2v-14B")
            cmd.extend(["--task", task])
            
            # Required model paths
            if inference_config.get("i2v_dit_path"):
                cmd.extend(["--dit", inference_config["i2v_dit_path"]])
            else:
                raise ValueError("DiT model path is required for I2V inference")
            
            # WAN 2.2 High Noise Model (if version is 2.2 and path provided)
            wan_version = inference_config.get("i2v_wan_version", "2.1")
            if wan_version == "2.2" and inference_config.get("i2v_dit_high_noise_path"):
                cmd.extend(["--dit_high_noise", inference_config["i2v_dit_high_noise_path"]])
            
            if inference_config.get("i2v_vae_path"):
                cmd.extend(["--vae", inference_config["i2v_vae_path"]])
            else:
                raise ValueError("VAE model path is required for I2V inference")
            
            if inference_config.get("i2v_t5_path"):
                cmd.extend(["--t5", inference_config["i2v_t5_path"]])
            else:
                raise ValueError("T5 text encoder path is required for I2V inference")
            
            # CLIP model (required for WAN 2.1, not for WAN 2.2)
            if wan_version == "2.1":
                if inference_config.get("i2v_clip_path"):
                    cmd.extend(["--clip", inference_config["i2v_clip_path"]])
                else:
                    raise ValueError("CLIP model path is required for WAN 2.1")
            elif wan_version == "2.2" and inference_config.get("i2v_clip_path"):
                # CLIP is optional for WAN 2.2 but can be provided
                cmd.extend(["--clip", inference_config["i2v_clip_path"]])
            
            # LoRA (optional for 1-frame inference, but recommended for better results)
            if inference_config.get("i2v_lora_path"):
                cmd.extend(["--lora_weight", inference_config["i2v_lora_path"]])
                cmd.extend(["--lora_multiplier", str(inference_config.get("i2v_lora_multiplier", 1.0))])
            else:
                print("WARNING: No LoRA specified for 1-frame inference. Results may be very similar to input image with noise. Consider using a LoRA for better quality.")
            
            # Required inference inputs
            if inference_config.get("i2v_input_image"):
                cmd.extend(["--image_path", inference_config["i2v_input_image"]])
                cmd.extend(["--control_image_path", inference_config["i2v_input_image"]])
            else:
                raise ValueError("Input image is required for I2V inference")
            
            if inference_config.get("i2v_prompt"):
                cmd.extend(["--prompt", f'"{inference_config["i2v_prompt"]}"'])
            else:
                raise ValueError("Prompt is required for I2V inference")
            
            # Optional generation parameters
            if inference_config.get("i2v_negative_prompt"):
                cmd.extend(["--negative_prompt", f'"{inference_config["i2v_negative_prompt"]}"'])
            
            # Image dimensions
            height = int(inference_config.get("i2v_height", 576))
            width = int(inference_config.get("i2v_width", 384))
            cmd.extend(["--video_size", str(height), str(width)])
            
            # One-frame inference settings
            target_index = int(inference_config.get("i2v_target_index", 1))
            control_index = int(inference_config.get("i2v_control_index", 0))
            one_frame_setting = f"target_index={target_index},control_index={control_index}"
            cmd.extend(["--one_frame_inference", one_frame_setting])
            
            # Generation parameters
            cmd.extend(["--infer_steps", str(int(inference_config.get("i2v_steps", 25)))])
            cmd.extend(["--guidance_scale", str(inference_config.get("i2v_guidance_scale", 7.0))])
            
            # Seed
            if inference_config.get("i2v_seed", -1) != -1:
                cmd.extend(["--seed", str(int(inference_config["i2v_seed"]))])
            
            # Output settings
            output_dir = inference_config.get("inference_output_dir", "outputs/inference")
            cmd.extend(["--save_path", output_dir])
            output_type = inference_config.get("output_type", "images")
            cmd.extend(["--output_type", output_type])
            
            # Advanced Performance Options
            attn_mode = inference_config.get("attn_mode", "torch")
            if attn_mode != "torch":  # Only add flag if not default torch mode
                cmd.extend(["--attn_mode", attn_mode])
            
            if inference_config.get("blocks_to_swap", 0) > 0:
                cmd.extend(["--blocks_to_swap", str(int(inference_config["blocks_to_swap"]))])
            
            if inference_config.get("enable_compile", False):
                cmd.append("--compile")
            
            # FP8 Options
            if inference_config.get("fp8_base", False):
                cmd.append("--fp8")
            
            if inference_config.get("fp8_scaled", False):
                cmd.append("--fp8_scaled")
            
            if inference_config.get("fp8_fast", False):
                cmd.append("--fp8_fast")
            
            if inference_config.get("fp8_t5", False):
                cmd.append("--fp8_t5")
            
            # Force VAE to use float32 to match encode/decode methods
            cmd.extend(["--vae_dtype", "float32"])
            
            # WAN 2.2 Specific Options
            if wan_version == "2.2":
                if inference_config.get("timestep_boundary") is not None:
                    cmd.extend(["--timestep_boundary", str(inference_config["timestep_boundary"])])
                
                if inference_config.get("guidance_scale_high_noise") is not None:
                    cmd.extend(["--guidance_scale_high_noise", str(inference_config["guidance_scale_high_noise"])])
                
                if inference_config.get("offload_inactive_dit", False):
                    cmd.append("--offload_inactive_dit")
                
                if inference_config.get("lazy_loading", False):
                    cmd.append("--lazy_loading")
            
            # Memory Options
            if inference_config.get("vae_cache_cpu", False):
                cmd.append("--vae_cache_cpu")
            
            if inference_config.get("flow_shift") is not None:
                cmd.extend(["--flow_shift", str(inference_config["flow_shift"])])
            
            if inference_config.get("trim_tail_frames", 0) > 0:
                cmd.extend(["--trim_tail_frames", str(int(inference_config["trim_tail_frames"]))])
            
            return " ".join(cmd)
            
        except Exception as e:
            return f"Error building I2V command: {str(e)}"
    
    def build_video_inference_command(self, inference_config):
        """Build standard video generation command with comprehensive WAN support"""
        try:
            venv_python = ".venv\\Scripts\\python.exe"
            cmd = [venv_python, "src/musubi_tuner/wan_generate_video.py"]
            
            # Task
            task = inference_config.get("video_task", "t2v-14B")
            cmd.extend(["--task", task])
            
            # Required model paths
            if inference_config.get("video_dit_path"):
                cmd.extend(["--dit", inference_config["video_dit_path"]])
            else:
                raise ValueError("DiT model path is required for video generation")
            
            # WAN 2.2 High Noise Model (if version is 2.2 and path provided)
            wan_version = inference_config.get("video_wan_version", "2.1")
            if wan_version == "2.2" and inference_config.get("video_dit_high_noise_path"):
                cmd.extend(["--dit_high_noise", inference_config["video_dit_high_noise_path"]])
            
            if inference_config.get("video_vae_path"):
                cmd.extend(["--vae", inference_config["video_vae_path"]])
            else:
                raise ValueError("VAE model path is required for video generation")
            
            if inference_config.get("video_t5_path"):
                cmd.extend(["--t5", inference_config["video_t5_path"]])
            else:
                raise ValueError("T5 text encoder path is required for video generation")
            
            # CLIP model (required for WAN 2.1, not for WAN 2.2)
            if wan_version == "2.1":
                if inference_config.get("video_clip_path"):
                    cmd.extend(["--clip", inference_config["video_clip_path"]])
                else:
                    raise ValueError("CLIP model path is required for WAN 2.1")
            elif wan_version == "2.2" and inference_config.get("video_clip_path"):
                # CLIP is optional for WAN 2.2 but can be provided
                cmd.extend(["--clip", inference_config["video_clip_path"]])
            
            # LoRA settings (optional for video generation)
            if inference_config.get("video_lora_path"):
                cmd.extend(["--lora_weight", inference_config["video_lora_path"]])
                cmd.extend(["--lora_multiplier", str(inference_config.get("video_lora_multiplier", 1.0))])
            
            # Input handling based on task
            if task in ["i2v-14B", "i2v-A14B", "flf2v-14B"]:
                if inference_config.get("video_input_image"):
                    cmd.extend(["--image_path", inference_config["video_input_image"]])
                else:
                    raise ValueError(f"Input image is required for {task}")
            
            # Required prompt
            if inference_config.get("video_prompt"):
                cmd.extend(["--prompt", f'"{inference_config["video_prompt"]}"'])
            else:
                raise ValueError("Prompt is required for video generation")
            
            # Optional negative prompt
            if inference_config.get("video_negative_prompt"):
                cmd.extend(["--negative_prompt", f'"{inference_config["video_negative_prompt"]}"'])
            
            # Video dimensions and length
            height = int(inference_config.get("video_height", 256))
            width = int(inference_config.get("video_width", 256))
            length = int(inference_config.get("video_length", 16))
            cmd.extend(["--video_size", str(height), str(width)])
            cmd.extend(["--video_length", str(length)])
            
            # Generation settings
            cmd.extend(["--infer_steps", str(int(inference_config.get("video_steps", 25)))])
            cmd.extend(["--guidance_scale", str(inference_config.get("video_guidance_scale", 7.0))])
            
            # Seed
            if inference_config.get("video_seed", -1) != -1:
                cmd.extend(["--seed", str(int(inference_config["video_seed"]))])
            
            # Output path
            output_dir = inference_config.get("inference_output_dir", "outputs/inference")
            cmd.extend(["--save_path", output_dir])
            output_type = inference_config.get("output_type", "video")
            cmd.extend(["--output_type", output_type])
            
            # Advanced Performance Options
            attn_mode = inference_config.get("attn_mode", "torch")
            if attn_mode != "torch":  # Only add flag if not default torch mode
                cmd.extend(["--attn_mode", attn_mode])
            
            if inference_config.get("blocks_to_swap", 0) > 0:
                cmd.extend(["--blocks_to_swap", str(int(inference_config["blocks_to_swap"]))])
            
            if inference_config.get("enable_compile", False):
                cmd.append("--compile")
            
            # FP8 Options
            if inference_config.get("fp8_base", False):
                cmd.append("--fp8")
            
            if inference_config.get("fp8_scaled", False):
                cmd.append("--fp8_scaled")
            
            if inference_config.get("fp8_fast", False):
                cmd.append("--fp8_fast")
            
            if inference_config.get("fp8_t5", False):
                cmd.append("--fp8_t5")
            
            # Force VAE to use float32 to match encode/decode methods
            cmd.extend(["--vae_dtype", "float32"])
            
            # WAN 2.2 Specific Options
            if wan_version == "2.2":
                if inference_config.get("timestep_boundary") is not None:
                    cmd.extend(["--timestep_boundary", str(inference_config["timestep_boundary"])])
                
                if inference_config.get("guidance_scale_high_noise") is not None:
                    cmd.extend(["--guidance_scale_high_noise", str(inference_config["guidance_scale_high_noise"])])
                
                if inference_config.get("offload_inactive_dit", False):
                    cmd.append("--offload_inactive_dit")
                
                if inference_config.get("lazy_loading", False):
                    cmd.append("--lazy_loading")
            
            # Memory Options
            if inference_config.get("vae_cache_cpu", False):
                cmd.append("--vae_cache_cpu")
            
            if inference_config.get("flow_shift") is not None:
                cmd.extend(["--flow_shift", str(inference_config["flow_shift"])])
            
            if inference_config.get("trim_tail_frames", 0) > 0:
                cmd.extend(["--trim_tail_frames", str(int(inference_config["trim_tail_frames"]))])
            
            return " ".join(cmd)
            
        except Exception as e:
            return f"Error building video command: {str(e)}"
    
    def build_vae_decode_command(self, latent_path, base_inference_config):
        """Build VAE decode command for debugging latent decoding"""
        try:
            if not latent_path or not latent_path.strip():
                return "Error: Please specify a latent file path"
            
            venv_python = ".venv\\Scripts\\python.exe"
            cmd = [venv_python, "src/musubi_tuner/wan_generate_video.py"]
            
            # Use latent decode mode
            cmd.extend(["--latent_path", latent_path.strip()])
            
            # Copy essential model paths from base config
            if base_inference_config.get("i2v_vae_path"):
                cmd.extend(["--vae", base_inference_config["i2v_vae_path"]])
            elif base_inference_config.get("video_vae_path"):
                cmd.extend(["--vae", base_inference_config["video_vae_path"]])
            else:
                return "Error: VAE model path is required for latent decoding"
            
            # Determine task from base config or default
            task = base_inference_config.get("i2v_task") or base_inference_config.get("video_task", "i2v-14B")
            cmd.extend(["--task", task])
            
            # Output settings
            output_dir = base_inference_config.get("inference_output_dir", "outputs/inference")
            cmd.extend(["--save_path", output_dir])
            cmd.extend(["--output_type", "images"])  # Always output images for decode
            
            # Force VAE to use float32 to match encode/decode methods
            cmd.extend(["--vae_dtype", "float32"])
            
            # Add debugging metadata preservation
            # cmd.extend(["--no_metadata"])  # Uncomment to disable metadata if needed
            
            return " ".join(cmd)
            
        except Exception as e:
            return f"Error building VAE decode command: {str(e)}"
    
    def inspect_latent_values(self, latent_path):
        """Inspect latent file values for debugging"""
        try:
            if not latent_path or not latent_path.strip():
                return "Error: Please specify a latent file path"
            
            import torch
            from safetensors.torch import load_file
            import os
            
            latent_file = latent_path.strip()
            if not os.path.exists(latent_file):
                return f"Error: Latent file not found: {latent_file}"
            
            # Load latent data
            data = load_file(latent_file)
            latent = data.get('latent', None)
            
            if latent is None:
                available_keys = list(data.keys())
                return f"Error: No 'latent' key found. Available keys: {available_keys}"
            
            # Calculate statistics
            shape = latent.shape
            dtype = latent.dtype
            mean_val = latent.mean().item()
            std_val = latent.std().item()
            min_val = latent.min().item()
            max_val = latent.max().item()
            
            # Check for unusual patterns
            zero_ratio = (latent == 0).float().mean().item()
            nan_count = torch.isnan(latent).sum().item()
            inf_count = torch.isinf(latent).sum().item()
            
            # Check if values are stuck at 0.5 or other suspicious patterns
            half_ratio = (torch.abs(latent - 0.5) < 0.01).float().mean().item()
            
            report = f"""Latent File Analysis: {os.path.basename(latent_file)}
═══════════════════════════════════════════════════════════
Shape: {shape}
Data Type: {dtype}
Mean: {mean_val:.6f}
Std Dev: {std_val:.6f}
Min: {min_val:.6f}
Max: {max_val:.6f}

Quality Checks:
• Zero values: {zero_ratio*100:.2f}% of values
• NaN values: {nan_count} count
• Inf values: {inf_count} count
• Values near 0.5: {half_ratio*100:.2f}% (suspicious if high)

Value Distribution (percentiles):
• 1%: {torch.quantile(latent, 0.01).item():.6f}
• 25%: {torch.quantile(latent, 0.25).item():.6f}
• 50%: {torch.quantile(latent, 0.50).item():.6f}
• 75%: {torch.quantile(latent, 0.75).item():.6f}
• 99%: {torch.quantile(latent, 0.99).item():.6f}

Diagnosis:
"""
            
            # Add diagnostic information
            if zero_ratio > 0.5:
                report += "⚠️ WARNING: >50% zero values - possible generation failure\n"
            if half_ratio > 0.3:
                report += "⚠️ WARNING: Many values near 0.5 - possible stuck activation\n"
            if std_val < 0.1:
                report += "⚠️ WARNING: Very low variance - possible collapsed generation\n"
            if nan_count > 0 or inf_count > 0:
                report += "❌ ERROR: NaN/Inf values detected - numerical instability\n"
            if abs(mean_val) < 0.01 and std_val > 0.5:
                report += "✅ GOOD: Normal-looking latent distribution\n"
            
            return report
            
        except Exception as e:
            return f"Error inspecting latent values: {str(e)}"
    
    def debug_vae_preprocessing(self, base_inference_config):
        """Debug VAE preprocessing and model info"""
        try:
            # Get VAE path from config
            vae_path = (base_inference_config.get("i2v_vae_path") or 
                       base_inference_config.get("video_vae_path"))
            
            if not vae_path:
                return "Error: No VAE model path found in configuration"
            
            report = f"""VAE Model & Preprocessing Debug
═══════════════════════════════════════════════════════════
VAE Model Path: {vae_path}
VAE Data Type: float32 (forced for consistency)

Expected VAE Preprocessing:
• Input range: [-1, 1] (from image pixel values [0, 255])
• Normalization: (pixel / 127.5) - 1.0
• Channel order: RGB
• VAE scaling factors:
  - Mean: [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, ...]
  - Std:  [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, ...]

Common Issues & Solutions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Gray Output:
   • Check if latent values are mostly zeros/constant
   • Verify VAE dtype consistency (should be float32)
   • Check if guidance scale is too high/low
   • Verify timestep boundary settings

2. VAE Decode Issues:
   • Ensure VAE weights are loaded correctly
   • Check autocast dtype matches VAE dtype
   • Verify latent scaling is applied correctly

3. Preprocessing Issues:
   • Input images should be normalized to [-1, 1]
   • VAE expects RGB channel order
   • Check image resizing doesn't corrupt data

Debug Commands:
• Use output_type='both' to save both latents and images
• Use output_type='latent' to inspect raw latents
• Compare latent statistics between working/broken generations
• Check if different guidance scales affect output

Recommended Settings for Debugging:
• guidance_scale: 3.5 (I2V), 3.0-4.0 (T2V)
• timestep_boundary: 0.900 (I2V), 0.875 (T2V)
• flow_shift: 5.0 (I2V), 12.0 (T2V)
• VAE dtype: float32
• Inference steps: 20-30
"""
            
            return report
            
        except Exception as e:
            return f"Error generating VAE debug info: {str(e)}"
    
    def build_training_command(self, model_config, training_config):
        """Build the training command from configuration"""
        try:
            # run in venv
            accelerate_venv = ".venv\\Scripts\\accelerate.exe"
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
                
                with gr.Tab("Inference"):
                    inference_components = self.create_inference_ui()
            
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
                    venv_python = ".venv\\Scripts\\python.exe"
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
                    venv_python = ".venv\\Scripts\\python.exe"
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
            
            # Inference event handlers
            def generate_i2v_inference(*args):
                """Generate I2V one-frame inference command and execute"""
                print("DEBUG: generate_i2v_inference called")
                print(f"DEBUG: Received {len(args)} arguments")
                
                try:
                    # Extract inference configuration from ALL components
                    inference_config = {
                        # I2V components (indices 0-18)
                        "i2v_wan_version": args[0],
                        "i2v_task": args[1],
                        "i2v_dit_path": args[2],
                        "i2v_dit_high_noise_path": args[3],
                        "i2v_vae_path": args[4],
                        "i2v_t5_path": args[5],
                        "i2v_clip_path": args[6],
                        "i2v_lora_path": args[7],
                        "i2v_lora_multiplier": args[8],
                        "i2v_input_image": args[9],
                        "i2v_prompt": args[10],
                        "i2v_negative_prompt": args[11],
                        "i2v_width": args[12],
                        "i2v_height": args[13],
                        "i2v_steps": args[14],
                        "i2v_guidance_scale": args[15],
                        "i2v_target_index": args[16],
                        "i2v_control_index": args[17],
                        "i2v_seed": args[18],
                        # Video components (indices 19-36) - not used for I2V but included for completeness
                        "video_wan_version": args[19],
                        "video_task": args[20],
                        "video_dit_path": args[21],
                        "video_dit_high_noise_path": args[22],
                        "video_vae_path": args[23],
                        "video_t5_path": args[24],
                        "video_clip_path": args[25],
                        "video_lora_path": args[26],
                        "video_lora_multiplier": args[27],
                        "video_input_image": args[28],
                        "video_prompt": args[29],
                        "video_negative_prompt": args[30],
                        "video_width": args[31],
                        "video_height": args[32],
                        "video_length": args[33],
                        "video_steps": args[34],
                        "video_guidance_scale": args[35],
                        "video_seed": args[36],
                        # Advanced options (indices 37-49)
                        "attn_mode": args[37],
                        "blocks_to_swap": args[38],
                        "enable_compile": args[39],
                        "fp8_base": args[40],
                        "fp8_scaled": args[41],
                        "fp8_fast": args[42],
                        "fp8_t5": args[43],
                        "timestep_boundary": args[44],
                        "guidance_scale_high_noise": args[45],
                        "offload_inactive_dit": args[46],
                        "lazy_loading": args[47],
                        "vae_cache_cpu": args[48],
                        "flow_shift": args[49],
                        "trim_tail_frames": args[50],
                        # Common components
                        "inference_output_dir": args[51],
                        "output_type": args[52]
                    }
                    
                    print(f"DEBUG: I2V config extracted - prompt: '{inference_config['i2v_prompt']}', task: '{inference_config['i2v_task']}'")
                    
                    # Build command
                    print("DEBUG: Building I2V inference command...")
                    command = self.build_i2v_inference_command(inference_config)
                    print(f"DEBUG: Generated command: {command}")
                    
                    # Check if command generation failed
                    if not command or command.startswith("Error"):
                        error_msg = f"Failed to generate I2V command: {command}"
                        print(f"DEBUG: {error_msg}")
                        return error_msg, "Error", error_msg
                    
                    # Start inference in background
                    def run_inference():
                        try:
                            print(f"DEBUG: Starting I2V inference process with command: {command}")
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                universal_newlines=True
                            )
                            
                            print(f"DEBUG: I2V inference process started with PID: {process.pid}")
                            
                            output_lines = []
                            for line in process.stdout:
                                line = line.strip()
                                if line:  # Only add non-empty lines
                                    output_lines.append(line)
                                    print(f"DEBUG: I2V output: {line}")
                                    if len(output_lines) > 100:  # Keep only last 100 lines
                                        output_lines = output_lines[-100:]
                            
                            return_code = process.wait()
                            print(f"DEBUG: I2V inference completed with return code: {return_code}")
                            
                            if return_code == 0:
                                final_output = "\n".join(output_lines) + "\n\n=== I2V inference completed successfully ==="
                                print("DEBUG: I2V inference completed successfully")
                            else:
                                final_output = "\n".join(output_lines) + f"\n\n=== I2V inference failed with exit code {return_code} ==="
                                print(f"DEBUG: I2V inference failed with exit code {return_code}")
                                
                        except Exception as e:
                            final_output = f"Error during I2V inference: {str(e)}"
                            print(f"DEBUG: Exception in I2V inference: {str(e)}")
                    
                    # Start inference thread
                    print("DEBUG: Starting I2V inference thread...")
                    inference_thread = threading.Thread(target=run_inference, daemon=True)
                    inference_thread.start()
                    
                    return command, "I2V inference started...", "I2V inference in progress..."
                    
                except Exception as e:
                    error_msg = f"Error starting I2V inference: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg, "Error", error_msg
            
            def generate_video_inference(*args):
                """Generate video inference command and execute"""
                print("DEBUG: generate_video_inference called")
                print(f"DEBUG: Received {len(args)} arguments")
                
                try:
                    # Extract video inference configuration from ALL components
                    inference_config = {
                        # I2V components (indices 0-18) - not used for video but included for completeness
                        "i2v_wan_version": args[0],
                        "i2v_task": args[1],
                        "i2v_dit_path": args[2],
                        "i2v_dit_high_noise_path": args[3],
                        "i2v_vae_path": args[4],
                        "i2v_t5_path": args[5],
                        "i2v_clip_path": args[6],
                        "i2v_lora_path": args[7],
                        "i2v_lora_multiplier": args[8],
                        "i2v_input_image": args[9],
                        "i2v_prompt": args[10],
                        "i2v_negative_prompt": args[11],
                        "i2v_width": args[12],
                        "i2v_height": args[13],
                        "i2v_steps": args[14],
                        "i2v_guidance_scale": args[15],
                        "i2v_target_index": args[16],
                        "i2v_control_index": args[17],
                        "i2v_seed": args[18],
                        # Video components (indices 19-36) - primary for video generation
                        "video_wan_version": args[19],
                        "video_task": args[20],
                        "video_dit_path": args[21],
                        "video_dit_high_noise_path": args[22],
                        "video_vae_path": args[23],
                        "video_t5_path": args[24],
                        "video_clip_path": args[25],
                        "video_lora_path": args[26],
                        "video_lora_multiplier": args[27],
                        "video_input_image": args[28],
                        "video_prompt": args[29],
                        "video_negative_prompt": args[30],
                        "video_width": args[31],
                        "video_height": args[32],
                        "video_length": args[33],
                        "video_steps": args[34],
                        "video_guidance_scale": args[35],
                        "video_seed": args[36],
                        # Advanced options (indices 37-49)
                        "attn_mode": args[37],
                        "blocks_to_swap": args[38],
                        "enable_compile": args[39],
                        "fp8_base": args[40],
                        "fp8_scaled": args[41],
                        "fp8_fast": args[42],
                        "fp8_t5": args[43],
                        "timestep_boundary": args[44],
                        "guidance_scale_high_noise": args[45],
                        "offload_inactive_dit": args[46],
                        "lazy_loading": args[47],
                        "vae_cache_cpu": args[48],
                        "flow_shift": args[49],
                        "trim_tail_frames": args[50],
                        # Common components
                        "inference_output_dir": args[51],
                        "output_type": args[52]
                    }
                    
                    print(f"DEBUG: Video config extracted - prompt: '{inference_config['video_prompt']}', task: '{inference_config['video_task']}'")
                    
                    # Build command
                    print("DEBUG: Building video inference command...")
                    command = self.build_video_inference_command(inference_config)
                    print(f"DEBUG: Generated command: {command}")
                    
                    # Check if command generation failed
                    if not command or command.startswith("Error"):
                        error_msg = f"Failed to generate video command: {command}"
                        print(f"DEBUG: {error_msg}")
                        return error_msg, "Error", error_msg
                    
                    # Start inference in background
                    def run_inference():
                        try:
                            print(f"DEBUG: Starting video inference process with command: {command}")
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                universal_newlines=True
                            )
                            
                            print(f"DEBUG: Video inference process started with PID: {process.pid}")
                            
                            output_lines = []
                            for line in process.stdout:
                                line = line.strip()
                                if line:  # Only add non-empty lines
                                    output_lines.append(line)
                                    print(f"DEBUG: Video output: {line}")
                                    if len(output_lines) > 100:  # Keep only last 100 lines
                                        output_lines = output_lines[-100:]
                            
                            return_code = process.wait()
                            print(f"DEBUG: Video inference completed with return code: {return_code}")
                            
                            if return_code == 0:
                                final_output = "\n".join(output_lines) + "\n\n=== Video inference completed successfully ==="
                                print("DEBUG: Video inference completed successfully")
                            else:
                                final_output = "\n".join(output_lines) + f"\n\n=== Video inference failed with exit code {return_code} ==="
                                print(f"DEBUG: Video inference failed with exit code {return_code}")
                                
                        except Exception as e:
                            final_output = f"Error during video inference: {str(e)}"
                            print(f"DEBUG: Exception in video inference: {str(e)}")
                    
                    # Start inference thread
                    print("DEBUG: Starting video inference thread...")
                    inference_thread = threading.Thread(target=run_inference, daemon=True)
                    inference_thread.start()
                    
                    return command, "Video inference started...", "Video inference in progress..."
                    
                except Exception as e:
                    error_msg = f"Error starting video inference: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg, "Error", error_msg

            def generate_vae_decode_command(latent_path, *args):
                """Generate VAE decode command for debugging"""
                print("DEBUG: generate_vae_decode_command called")
                print(f"DEBUG: latent_path='{latent_path}', args_count={len(args)}")
                
                try:
                    # Extract basic inference configuration for VAE models
                    inference_config = {
                        # I2V components
                        "i2v_vae_path": args[3] if len(args) > 3 else "",
                        "i2v_task": args[1] if len(args) > 1 else "i2v-14B",
                        # Video components
                        "video_vae_path": args[22] if len(args) > 22 else "",
                        "video_task": args[19] if len(args) > 19 else "t2v-14B",
                        # Common settings
                        "inference_output_dir": args[48] if len(args) > 48 else "outputs/inference",
                    }
                    
                    # Generate VAE decode command
                    command = self.build_vae_decode_command(latent_path, inference_config)
                    return command
                    
                except Exception as e:
                    error_msg = f"Error generating VAE decode command: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg

            def inspect_latent_values_callback(latent_path):
                """Callback to inspect latent values"""
                print("DEBUG: inspect_latent_values_callback called")
                print(f"DEBUG: latent_path='{latent_path}'")
                
                try:
                    return self.inspect_latent_values(latent_path)
                except Exception as e:
                    error_msg = f"Error inspecting latent values: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg

            def debug_vae_stats_callback(*args):
                """Callback to generate VAE preprocessing debug info"""
                print("DEBUG: debug_vae_stats_callback called")
                print(f"DEBUG: args_count={len(args)}")
                
                try:
                    # Extract basic inference configuration for VAE models
                    inference_config = {
                        # I2V components
                        "i2v_vae_path": args[3] if len(args) > 3 else "",
                        "i2v_task": args[1] if len(args) > 1 else "i2v-14B",
                        # Video components
                        "video_vae_path": args[22] if len(args) > 22 else "",
                        "video_task": args[19] if len(args) > 19 else "t2v-14B",
                    }
                    
                    return self.debug_vae_preprocessing(inference_config)
                except Exception as e:
                    error_msg = f"Error generating VAE debug info: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg

            # Inference Configuration Management Functions
            def save_inference_configuration(config_name, *args):
                """Save inference configuration to file"""
                print(f"DEBUG: save_inference_configuration called with config_name='{config_name}', args_count={len(args)}")
                
                try:
                    if not config_name or config_name.strip() == "":
                        return "Please enter a configuration name"
                    
                    # Extract configuration from ALL components
                    inference_config = {
                        # I2V components
                        "i2v_wan_version": args[0] if len(args) > 0 else "2.1",
                        "i2v_task": args[1] if len(args) > 1 else "i2v-14B",
                        "i2v_dit_path": args[2] if len(args) > 2 else "",
                        "i2v_dit_high_noise_path": args[3] if len(args) > 3 else "",
                        "i2v_vae_path": args[4] if len(args) > 4 else "",
                        "i2v_t5_path": args[5] if len(args) > 5 else "",
                        "i2v_clip_path": args[6] if len(args) > 6 else "",
                        "i2v_lora_path": args[7] if len(args) > 7 else "",
                        "i2v_lora_multiplier": args[8] if len(args) > 8 else 1.0,
                        "i2v_input_image": args[9] if len(args) > 9 else None,
                        "i2v_prompt": args[10] if len(args) > 10 else "",
                        "i2v_negative_prompt": args[11] if len(args) > 11 else "",
                        "i2v_width": args[12] if len(args) > 12 else 384,
                        "i2v_height": args[13] if len(args) > 13 else 576,
                        "i2v_steps": args[14] if len(args) > 14 else 25,
                        "i2v_guidance_scale": args[15] if len(args) > 15 else 7.0,
                        "i2v_target_index": args[16] if len(args) > 16 else 1,
                        "i2v_control_index": args[17] if len(args) > 17 else 0,
                        "i2v_seed": args[18] if len(args) > 18 else -1,
                        # Video components
                        "video_wan_version": args[19] if len(args) > 19 else "2.1",
                        "video_task": args[20] if len(args) > 20 else "t2v-14B",
                        "video_dit_path": args[21] if len(args) > 21 else "",
                        "video_dit_high_noise_path": args[22] if len(args) > 22 else "",
                        "video_vae_path": args[23] if len(args) > 23 else "",
                        "video_t5_path": args[24] if len(args) > 24 else "",
                        "video_clip_path": args[25] if len(args) > 25 else "",
                        "video_lora_path": args[26] if len(args) > 26 else "",
                        "video_lora_multiplier": args[27] if len(args) > 27 else 1.0,
                        "video_input_image": args[28] if len(args) > 28 else None,
                        "video_prompt": args[29] if len(args) > 29 else "",
                        "video_negative_prompt": args[30] if len(args) > 30 else "",
                        "video_width": args[31] if len(args) > 31 else 256,
                        "video_height": args[32] if len(args) > 32 else 256,
                        "video_length": args[33] if len(args) > 33 else 16,
                        "video_steps": args[34] if len(args) > 34 else 25,
                        "video_guidance_scale": args[35] if len(args) > 35 else 7.0,
                        "video_seed": args[36] if len(args) > 36 else -1,
                        # Advanced options
                        "attn_mode": args[37] if len(args) > 37 else "torch",
                        "blocks_to_swap": args[38] if len(args) > 38 else 0,
                        "enable_compile": args[39] if len(args) > 39 else False,
                        "fp8_base": args[40] if len(args) > 40 else False,
                        "fp8_scaled": args[41] if len(args) > 41 else False,
                        "fp8_fast": args[42] if len(args) > 42 else False,
                        "fp8_t5": args[43] if len(args) > 43 else False,
                        "timestep_boundary": args[44] if len(args) > 44 else 0.875,
                        "guidance_scale_high_noise": args[45] if len(args) > 45 else 7.0,
                        "offload_inactive_dit": args[46] if len(args) > 46 else False,
                        "lazy_loading": args[47] if len(args) > 47 else False,
                        "vae_cache_cpu": args[48] if len(args) > 48 else False,
                        "flow_shift": args[49] if len(args) > 49 else 3.0,
                        "trim_tail_frames": args[50] if len(args) > 50 else 0,
                        # Common components
                        "inference_output_dir": args[51] if len(args) > 51 else "outputs/inference",
                        "output_type": args[52] if len(args) > 52 else "images"
                    }
                    
                    # Save configuration to file
                    import os
                    import json
                    
                    configs_dir = "configs/inference"
                    os.makedirs(configs_dir, exist_ok=True)
                    
                    config_file = os.path.join(configs_dir, f"{config_name}.json")
                    with open(config_file, 'w') as f:
                        json.dump(inference_config, f, indent=2)
                    
                    return f"Inference configuration '{config_name}' saved successfully"
                    
                except Exception as e:
                    error_msg = f"Error saving inference configuration: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return error_msg

            def load_inference_configuration(config_name):
                """Load inference configuration from file"""
                print(f"DEBUG: load_inference_configuration called with config_name='{config_name}'")
                
                try:
                    if not config_name or config_name.strip() == "":
                        return [None] * 52 + ["Please select a configuration to load"]
                    
                    import os
                    import json
                    
                    config_file = os.path.join("configs/inference", f"{config_name}.json")
                    
                    if not os.path.exists(config_file):
                        return [None] * 52 + [f"Configuration '{config_name}' not found"]
                    
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Return all values in the correct order for the UI components
                    return [
                        # I2V components
                        config.get("i2v_wan_version", "2.1"),
                        config.get("i2v_task", "i2v-14B"),
                        config.get("i2v_dit_path", ""),
                        config.get("i2v_dit_high_noise_path", ""),
                        config.get("i2v_vae_path", ""),
                        config.get("i2v_t5_path", ""),
                        config.get("i2v_clip_path", ""),
                        config.get("i2v_lora_path", ""),
                        config.get("i2v_lora_multiplier", 1.0),
                        config.get("i2v_input_image", None),
                        config.get("i2v_prompt", ""),
                        config.get("i2v_negative_prompt", ""),
                        config.get("i2v_width", 384),
                        config.get("i2v_height", 576),
                        config.get("i2v_steps", 25),
                        config.get("i2v_guidance_scale", 7.0),
                        config.get("i2v_target_index", 1),
                        config.get("i2v_control_index", 0),
                        config.get("i2v_seed", -1),
                        # Video components
                        config.get("video_wan_version", "2.1"),
                        config.get("video_task", "t2v-14B"),
                        config.get("video_dit_path", ""),
                        config.get("video_dit_high_noise_path", ""),
                        config.get("video_vae_path", ""),
                        config.get("video_t5_path", ""),
                        config.get("video_clip_path", ""),
                        config.get("video_lora_path", ""),
                        config.get("video_lora_multiplier", 1.0),
                        config.get("video_input_image", None),
                        config.get("video_prompt", ""),
                        config.get("video_negative_prompt", ""),
                        config.get("video_width", 256),
                        config.get("video_height", 256),
                        config.get("video_length", 16),
                        config.get("video_steps", 25),
                        config.get("video_guidance_scale", 7.0),
                        config.get("video_seed", -1),
                        # Advanced options
                        config.get("attn_mode", "torch"),
                        config.get("blocks_to_swap", 0),
                        config.get("enable_compile", False),
                        config.get("fp8_base", False),
                        config.get("fp8_scaled", False),
                        config.get("fp8_fast", False),
                        config.get("fp8_t5", False),
                        config.get("timestep_boundary", 0.875),
                        config.get("guidance_scale_high_noise", 7.0),
                        config.get("offload_inactive_dit", False),
                        config.get("lazy_loading", False),
                        config.get("vae_cache_cpu", False),
                        config.get("flow_shift", 3.0),
                        config.get("trim_tail_frames", 0),
                        # Common components
                        config.get("inference_output_dir", "outputs/inference"),
                        config.get("output_type", "images")
                    ] + [f"Configuration '{config_name}' loaded successfully"]
                    
                except Exception as e:
                    error_msg = f"Error loading inference configuration: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    return [None] * 53 + [error_msg]  # Return empty values for all components + error message

            def delete_inference_configuration(config_name):
                """Delete inference configuration file"""
                try:
                    if not config_name or config_name.strip() == "":
                        return "Please select a configuration to delete"
                    
                    import os
                    
                    config_file = os.path.join("configs/inference", f"{config_name}.json")
                    
                    if not os.path.exists(config_file):
                        return f"Configuration '{config_name}' not found"
                    
                    os.remove(config_file)
                    return f"Configuration '{config_name}' deleted successfully"
                    
                except Exception as e:
                    return f"Error deleting configuration: {str(e)}"

            def refresh_inference_configs():
                """Refresh the list of available inference configurations"""
                try:
                    import os
                    
                    configs_dir = "configs/inference"
                    if not os.path.exists(configs_dir):
                        return gr.Dropdown(choices=[])
                    
                    config_files = [f[:-5] for f in os.listdir(configs_dir) if f.endswith('.json')]
                    config_files.sort()
                    
                    return gr.Dropdown(choices=config_files)
                    
                except Exception as e:
                    print(f"Error refreshing inference configs: {str(e)}")
                    return gr.Dropdown(choices=[])

            # Get inference component inputs - ALL components for complete config saving
            inference_inputs = [
                # I2V inputs (20 components, indices 0-19)
                inference_components["i2v_wan_version"],
                inference_components["i2v_task"],
                inference_components["i2v_dit_path"],
                inference_components["i2v_dit_high_noise_path"],
                inference_components["i2v_vae_path"],
                inference_components["i2v_t5_path"],
                inference_components["i2v_clip_path"],
                inference_components["i2v_lora_path"],
                inference_components["i2v_lora_multiplier"],
                inference_components["i2v_input_image"],
                inference_components["i2v_prompt"],
                inference_components["i2v_negative_prompt"],
                inference_components["i2v_width"],
                inference_components["i2v_height"],
                inference_components["i2v_steps"],
                inference_components["i2v_guidance_scale"],
                inference_components["i2v_target_index"],
                inference_components["i2v_control_index"],
                inference_components["i2v_seed"],
                # Video inputs (18 components, indices 19-36)
                inference_components["video_wan_version"],
                inference_components["video_task"],
                inference_components["video_dit_path"],
                inference_components["video_dit_high_noise_path"],
                inference_components["video_vae_path"],
                inference_components["video_t5_path"],
                inference_components["video_clip_path"],
                inference_components["video_lora_path"],
                inference_components["video_lora_multiplier"],
                inference_components["video_input_image"],
                inference_components["video_prompt"],
                inference_components["video_negative_prompt"],
                inference_components["video_width"],
                inference_components["video_height"],
                inference_components["video_length"],
                inference_components["video_steps"],
                inference_components["video_guidance_scale"],
                inference_components["video_seed"],
                # Advanced options (12 components, indices 37-48)
                inference_components["attn_mode"],
                inference_components["blocks_to_swap"],
                inference_components["enable_compile"],
                inference_components["fp8_base"],
                inference_components["fp8_scaled"],
                inference_components["fp8_fast"],
                inference_components["fp8_t5"],
                inference_components["timestep_boundary"],
                inference_components["guidance_scale_high_noise"],
                inference_components["offload_inactive_dit"],
                inference_components["lazy_loading"],
                inference_components["vae_cache_cpu"],
                inference_components["flow_shift"],
                inference_components["trim_tail_frames"],
                # Common components (2 components, indices 51-52)
                inference_components["inference_output_dir"],
                inference_components["output_type"]
            ]
            
            # Connect inference event handlers
            inference_components["generate_i2v_btn"].click(
                generate_i2v_inference,
                inputs=inference_inputs,
                outputs=[
                    inference_components["inference_command_output"],
                    inference_components["inference_status"],
                    inference_components["inference_output"]
                ]
            )
            print("DEBUG: Generate I2V inference button event handler connected")
            
            inference_components["generate_video_btn"].click(
                generate_video_inference,
                inputs=inference_inputs,
                outputs=[
                    inference_components["inference_command_output"],
                    inference_components["inference_status"],
                    inference_components["inference_output"]
                ]
            )
            print("DEBUG: Generate video inference button event handler connected")
            
            inference_components["generate_vae_decode_btn"].click(
                generate_vae_decode_command,
                inputs=[inference_components["latent_file_path"]] + inference_inputs,
                outputs=inference_components["vae_decode_command_output"]
            )
            print("DEBUG: Generate VAE decode command button event handler connected")
            
            inference_components["inspect_latents_btn"].click(
                inspect_latent_values_callback,
                inputs=inference_components["latent_file_path"],
                outputs=inference_components["latent_debug_output"]
            )
            print("DEBUG: Inspect latents button event handler connected")
            
            inference_components["debug_vae_stats_btn"].click(
                debug_vae_stats_callback,
                inputs=inference_inputs,
                outputs=inference_components["latent_debug_output"]
            )
            print("DEBUG: Debug VAE stats button event handler connected")
            
            # Connect inference configuration management event handlers
            # All inference config components for input (name + all components)
            inference_config_inputs = [inference_components["inference_config_name"]] + inference_inputs
            
            inference_components["save_inference_config_btn"].click(
                save_inference_configuration,
                inputs=inference_config_inputs,
                outputs=[inference_components["inference_config_status"]]
            )
            
            inference_components["load_inference_config_btn"].click(
                load_inference_configuration,
                inputs=[inference_components["inference_config_list"]],
                outputs=inference_inputs + [inference_components["inference_config_status"]]
            )
            
            inference_components["delete_inference_config_btn"].click(
                delete_inference_configuration,
                inputs=[inference_components["inference_config_list"]],
                outputs=[inference_components["inference_config_status"]]
            )
            
            inference_components["refresh_inference_configs_btn"].click(
                refresh_inference_configs,
                outputs=[inference_components["inference_config_list"]]
            )
            
            print("DEBUG: Inference configuration management button event handlers connected")
        
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
