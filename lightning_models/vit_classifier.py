import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.vit import VisionTransformer
from torch.optim.lr_scheduler import LambdaLR
import math
from config import BaseConfig

class ViTClassifier(pl.LightningModule):
    def __init__(self, model: VisionTransformer, config: BaseConfig):
        super().__init__()
        self.model = model
        self.lr = float(config.learning_rate) * config.gpus
        self.warmup_steps = config.warmup_steps
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        def lr_lambda(step):
            # Linear warmup followed by cosine decay
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(max(1, self.trainer.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        } 