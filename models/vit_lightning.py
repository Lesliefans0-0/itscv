import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.vit import VisionTransformer
from models.iterative_vit import IterativeViT
from models.recursive_vit import RecursiveViT

# --- LightningModule ---
class ViTClassifier(pl.LightningModule):
    def __init__(self, model: VisionTransformer, lr=3e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _calculate_metrics(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._calculate_metrics(batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._calculate_metrics(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._calculate_metrics(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class IterativeViTClassifier(ViTClassifier):
    def __init__(self, model: IterativeViT, lr=3e-4):
        super().__init__(model, lr)
        self.num_iterations = model.num_iterations

    def _calculate_metrics_for_iterations(self, batch):
        x, y = batch
        outputs = self(x) # This will be a list of logits for IterativeViT

        total_loss = 0
        losses = []
        accs = []
        for logits in outputs:
            loss = self.criterion(logits, y)
            # Average the loss across iterations for the main loss metric
            total_loss += loss * (1 / self.num_iterations) 
            losses.append(loss)
            accs.append((logits.argmax(dim=1) == y).float().mean())
            
        # Calculate accuracy based on the final iteration's output
        final_acc = accs[-1]
        
        return total_loss, final_acc, losses, accs

    def _calculate_metrics(self, batch):
        # This override is still needed for the inherited training_step
        total_loss, final_acc, _, _ = self._calculate_metrics_for_iterations(batch)
        return total_loss, final_acc

    def _log_step_metrics(self, batch, batch_idx, stage: str):
        """Calculates and logs metrics for a given stage (val/test)."""
        total_loss, final_acc, losses, accs = self._calculate_metrics_for_iterations(batch)
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_acc", final_acc, prog_bar=True) # Log final accuracy

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

class RecursiveViTClassifier(IterativeViTClassifier):
    def __init__(self, model: RecursiveViT, lr=3e-4):
        super().__init__(model, lr)
