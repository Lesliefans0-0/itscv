# main.py
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
from models.vit_lightning import ViTClassifier

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

# CUDA_VISIBLE_DEVICES=5,6 python main.py --experiment vit_iterative_2025-04-06-0134; 
# CUDA_VISIBLE_DEVICES=5,6 python main.py --experiment vit_base_2025-04-06-0134