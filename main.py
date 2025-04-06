# vit_lightning.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse

from models.vit import VisionTransformer
from models.iterative_vit import IterativeViT
from data.imagenet_datamodule import ImageNetDataModule, CustomImageNetDataModule
from config import get_experiment_config
from callbacks import TokenSimilarityCallback

# --- LightningModule ---
class ViTClassifier(pl.LightningModule):
    def __init__(self, model, lr=3e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# --- Training Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment config to use')
    args = parser.parse_args()

    # Load experiment config
    cfg = get_experiment_config(args.experiment)
    
    # Create experiment directories
    experiment_dir = os.path.join('experiments', cfg.experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Save the config used for this experiment
    cfg.save_config(experiment_dir)

    # Initialize data module
    data_module = CustomImageNetDataModule(
        imagenet_dir=cfg.imagenet_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    # Initialize model
    if cfg.use_iterative_vit:
        torch_model = IterativeViT(
            num_iterative_tokens=cfg.num_iterative_tokens,
            num_iterations=cfg.num_iterations,
            layer_idx=cfg.layer_idx,
            emb_size=cfg.emb_size
        )
    else:
        torch_model = VisionTransformer(
            emb_size=cfg.emb_size
        )

    # wrap the model with LightningModule
    model = ViTClassifier(model=torch_model, lr=float(cfg.learning_rate))
    
    # Setup logger
    logger = TensorBoardLogger(tensorboard_dir, name=cfg.experiment_name)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{cfg.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    token_similarity_callback = TokenSimilarityCallback(log_every_n_steps=100)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='gpu',
        devices=cfg.gpus,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
        precision=cfg.precision,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[checkpoint_callback, token_similarity_callback]
    )

    # Start training
    trainer.fit(model, data_module)

    # Start testing
    trainer.test(model, data_module)

# CUDA_VISIBLE_DEVICES=1,5,6 python main.py --experiment vit_iterative_2025-04-06-0134; 
# CUDA_VISIBLE_DEVICES=1,5,6 python main.py --experiment vit_base_2025-04-06-0134