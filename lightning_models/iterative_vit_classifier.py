import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.iterative_vit import IterativeViT
from .vit_classifier import ViTClassifier
from config import BaseConfig

class IterativeViTClassifier(ViTClassifier):
    def __init__(self, model: IterativeViT, config: BaseConfig):
        super().__init__(model, config)
        self.num_iterations = model.num_iterations
        self.loss_assignment = config.loss_assignment

    def _calculate_metrics_for_iterations(self, batch):
        x, y = batch
        outputs = self(x) # This will be a list of logits for IterativeViT

        total_loss = 0
        losses = []
        accs = []
        for logits in outputs:
            loss = self.criterion(logits, y)
            # Average the loss across iterations for the main loss metric
            if self.loss_assignment == 'all_iterations':
                total_loss += loss * (1 / self.num_iterations)
            losses.append(loss)
            accs.append((logits.argmax(dim=1) == y).float().mean())
            
        # Calculate accuracy based on the final iteration's output
        final_acc = accs[-1]
        if self.loss_assignment == 'last_iteration':
            total_loss = losses[-1]
        
        return total_loss, final_acc, losses, accs

    def _calculate_metrics(self, batch):
        # This override is still needed for the inherited training_step
        total_loss, final_acc, _, _ = self._calculate_metrics_for_iterations(batch)
        return total_loss, final_acc

    def _log_step_metrics(self, batch, batch_idx, stage: str):
        """Calculates and logs metrics for a given stage (val/test)."""
        total_loss, final_acc, losses, accs = self._calculate_metrics_for_iterations(batch)
        self.log(f"{stage}_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", final_acc, prog_bar=True, sync_dist=True) # Log final accuracy

        # Log accuracy for each iteration under a common tag, grouped by iteration
        for i, acc in enumerate(accs):
            self.log(f"{stage}_acc_per_iter/iter_{i}", acc, prog_bar=False, sync_dist=True)
        # Optionally log per-iteration losses as well
        for i, loss in enumerate(losses):
            self.log(f"{stage}_loss_per_iter/iter_{i}", loss, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self._log_step_metrics(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._log_step_metrics(batch, batch_idx, 'test')