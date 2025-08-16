"""Configuration management for musubi-tuner UI"""
import json
import os
import toml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages configuration files for training and dataset setups"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_train_config = {
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
                "network_module": "networks.lora_wan",
                "network_dim": 32,
                "network_alpha": None,
                "network_dropout": None,
                "mixed_precision": "bf16",
                "fp8_base": True,
                "fp8_scaled": False,
                "fp8_t5": False,
                "gradient_checkpointing": True,
                "max_train_epochs": 16,
                "save_every_n_epochs": 1,
                "seed": 42,
                "batch_size": 1,
                "max_data_loader_n_workers": 2,
                "persistent_data_loader_workers": True,
                "timestep_sampling": "shift",
                "discrete_flow_shift": 3.0,
                "min_timestep": None,
                "max_timestep": None,
                "preserve_distribution_shape": False,
                "blocks_to_swap": None,
                "output_dir": "outputs",
                "output_name": "wan_lora",
                "logging_dir": None,
                "log_with": None,
                "wandb_api_key": "",
                "wandb_run_name": "",
                "log_config": False
            },
            "memory": {
                "vae_cache_cpu": False,
                "sdpa": True,
                "split_attn": False,
                "flash_attn": False,
                "xformers": False,
                "sage_attn": False
            },
            "sampling": {
                "sample_prompts": "",
                "sample_every_n_epochs": None,
                "sample_every_n_steps": None,
                "sample_at_first": False,
                "one_frame": False
            }
        }
        
        self.default_dataset_config = {
            "general": {
                "resolution": [960, 544],
                "caption_extension": ".txt",
                "batch_size": 1,
                "enable_bucket": True,
                "bucket_no_upscale": False
            },
            "datasets": []
        }
        
        self.wan_tasks = {
            "t2v-1.3B": {"name": "WAN2.1 T2V 1.3B", "requires_clip": False, "version": "2.1"},
            "t2v-14B": {"name": "WAN2.1 T2V 14B", "requires_clip": False, "version": "2.1"},
            "i2v-14B": {"name": "WAN2.1 I2V 14B", "requires_clip": True, "version": "2.1"},
            "t2i-14B": {"name": "WAN2.1 T2I 14B", "requires_clip": True, "version": "2.1"},
            "t2v-1.3B-FC": {"name": "WAN2.1 T2V 1.3B Fun Control", "requires_clip": False, "version": "2.1"},
            "t2v-14B-FC": {"name": "WAN2.1 T2V 14B Fun Control", "requires_clip": False, "version": "2.1"},
            "i2v-14B-FC": {"name": "WAN2.1 I2V 14B Fun Control", "requires_clip": True, "version": "2.1"},
            "t2v-A14B": {"name": "WAN2.2 T2V 14B", "requires_clip": False, "version": "2.2"},
            "i2v-A14B": {"name": "WAN2.2 I2V 14B", "requires_clip": False, "version": "2.2"}
        }
        
        self.memory_attention_modes = [
            "sdpa", "flash_attn", "xformers", "sage_attn"
        ]
        
        self.timestep_sampling_modes = [
            "uniform", "shift", "logsnr", "qinglong_flux", "qinglong_qwen"
        ]
        
        self.optimizer_types = [
            "adamw", "adamw8bit", "lion", "lion8bit", "sgdnesterov", "sgdnesterov8bit",
            "dadaptation", "dadaptlion", "dadaptsgd", "dadaptadam", "dadaptadagrad", "dadaptadan"
        ]
    
    def save_config(self, config: Dict[str, Any], filename: str) -> None:
        """Save configuration to file"""
        config_path = self.config_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config {filename}: {e}")
            return None
    
    def list_configs(self) -> list:
        """List all available configuration files"""
        if not self.config_dir.exists():
            return []
        
        configs = []
        for file in self.config_dir.glob("*.json"):
            configs.append(file.stem)
        return sorted(configs)
    
    def delete_config(self, filename: str) -> bool:
        """Delete a configuration file"""
        config_path = self.config_dir / f"{filename}.json"
        if config_path.exists():
            config_path.unlink()
            return True
        return False
    
    def generate_dataset_toml(self, config: Dict[str, Any], output_path: str) -> None:
        """Generate TOML dataset configuration file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            toml.dump(config, f)
    
    def load_dataset_toml(self, toml_path: str) -> Optional[Dict[str, Any]]:
        """Load dataset configuration from TOML file"""
        try:
            return toml.load(toml_path)
        except Exception as e:
            print(f"Error loading TOML file {toml_path}: {e}")
            return None
    
    def get_task_info(self, task: str) -> Dict[str, Any]:
        """Get information about a specific task"""
        return self.wan_tasks.get(task, {})
    
    def get_recommended_settings_for_task(self, task: str) -> Dict[str, Any]:
        """Get recommended settings for a specific task"""
        task_info = self.get_task_info(task)
        recommended = {}
        
        if task_info.get("version") == "2.2":
            if "i2v" in task.lower():
                recommended["discrete_flow_shift"] = 5.0
                recommended["timestep_boundary"] = 0.9
            else:  # t2v
                recommended["discrete_flow_shift"] = 12.0
                recommended["timestep_boundary"] = 0.875
        else:  # WAN 2.1
            recommended["discrete_flow_shift"] = 3.0
            
        return recommended
    
    def validate_config(self, config: Dict[str, Any]) -> list:
        """Validate configuration and return list of issues"""
        issues = []
        
        model_config = config.get("model", {})
        
        # Check required paths
        if not model_config.get("dit_path"):
            issues.append("DiT model path is required")
        if not model_config.get("t5_path"):
            issues.append("T5 model path is required")
        if not model_config.get("vae_path"):
            issues.append("VAE model path is required")
            
        task = model_config.get("task", "")
        task_info = self.get_task_info(task)
        
        # Check CLIP requirement
        if task_info.get("requires_clip") and not model_config.get("clip_path"):
            issues.append(f"CLIP model path is required for task {task}")
            
        # Check WAN 2.2 specific requirements
        if task_info.get("version") == "2.2":
            if not model_config.get("dit_high_noise_path"):
                issues.append("High noise DiT model path is recommended for WAN 2.2")
                
        # Check training settings
        training_config = config.get("training", {})
        if training_config.get("learning_rate", 0) <= 0:
            issues.append("Learning rate must be greater than 0")
            
        return issues
