import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

class TokenSimilarityCallback(Callback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            # Get the model's outputs
            if hasattr(pl_module.model, 'all_outputs'):
                all_outputs = pl_module.model.all_outputs
                
                # Calculate cosine similarity between tokens for each iteration
                for iter_idx, output in enumerate(all_outputs):
                    # Normalize the tokens
                    normalized_tokens = F.normalize(output, dim=-1)
                    
                    # Calculate cosine similarity matrix
                    similarity_matrix = torch.matmul(normalized_tokens, normalized_tokens.transpose(-2, -1))
                    
                    # Log to tensorboard
                    for logger in trainer.loggers:
                        if isinstance(logger, TensorBoardLogger):
                            logger.experiment.add_image(
                                f'token_similarity/iteration_{iter_idx}',
                                similarity_matrix[0].unsqueeze(0),  # Take first sample
                                global_step=trainer.global_step
                            ) 