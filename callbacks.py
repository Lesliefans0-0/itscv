import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_models import ViTClassifier
from typing import List
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

class TokenSimilarityCallback(Callback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        self.layer_outputs = []
        
    def on_train_start(self, trainer, pl_module: ViTClassifier):
        # Register hooks for each transformer block
        def get_layer_hook(layer_idx):
            def hook(module, input, output):
                self.layer_outputs[layer_idx] = output
            return hook
            
        # Clear any existing hooks
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_outputs: List[torch.Tensor] = []
        
        # Register hooks for each transformer block
        for i, block in enumerate(pl_module.model.blocks):
            self.layer_outputs.append(None)
            hook = block.register_forward_hook(get_layer_hook(i))
            self.hooks.append(hook)
            
    def on_train_end(self, trainer, pl_module):
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = []
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: ViTClassifier, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            # For regular ViT, use layer outputs
            for layer_idx, layer_output in enumerate(self.layer_outputs):
                if layer_output is not None:
                    # Normalize the tokens
                    normalized_tokens = F.normalize(layer_output, dim=-1)
                    
                    # Calculate cosine similarity matrix
                    similarity_matrix = torch.matmul(normalized_tokens, normalized_tokens.transpose(-2, -1))
                    
                    # Log to tensorboard
                    for logger in trainer.loggers:
                        if isinstance(logger, TensorBoardLogger):
                            logger.experiment.add_image(
                                f'token_similarity/layer_{layer_idx}',
                                similarity_matrix[0].unsqueeze(0),  # Take first sample
                                global_step=trainer.global_step
                            )
