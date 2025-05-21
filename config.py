import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseConfig:
    # Data settings
    imagenet_dir: str = "/mnt/d1/eirwynliang/imagenet-1k-processed-deepseek"
    batch_size: int = 128
    num_workers: int = 8
    num_classes: int = 1000

    # Training settings
    gpus: int = 8
    max_epochs: int = 50
    warmup_steps: int = 5 * 1e4 # 10k per epoch
    learning_rate: float = 3e-4

    # Model settings
    precision: int = 16
    log_every_n_steps: int = 50
    emb_size: int = 768
    depth: int = 12

    # Experiment settings
    model_type: str = 'vit'
    num_iterative_tokens: int = 3
    num_iterations: int = 5
    layer_idx: int = -2
    loss_assignment: str = 'all_iterations' # 'last_iteration' or 'all_iterations'

    # Experiment name (will be used for logging and checkpoints)
    experiment_name: str = "default"

    # Calculated config, do not set
    

    @classmethod
    def from_yaml(cls, config_path: str) -> 'BaseConfig':
        """Load config from a YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_config(self, save_dir: str):
        """Save config to YAML file"""
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

def get_experiment_config(experiment_name: str) -> BaseConfig:
    """Load experiment config from configs directory"""
    config_path = os.path.join('configs', f'{experiment_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return BaseConfig.from_yaml(config_path)