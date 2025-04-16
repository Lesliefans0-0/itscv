import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.vit import VisionTransformer

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