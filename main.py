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
from models.recursive_vit import RecursiveViT
from data.imagenet_datamodule import ImageNetDataModule, CustomImageNetDataModule
from config import get_experiment_config
from callbacks import TokenSimilarityCallback
from lightning_models import ViTClassifier, IterativeViTClassifier, RecursiveViTClassifier

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
    if cfg.model_type == 'iterative_vit':
        torch_model = IterativeViT(
            num_iterative_tokens=cfg.num_iterative_tokens,
            num_iterations=cfg.num_iterations,
            layer_idx=cfg.layer_idx,
            emb_size=cfg.emb_size,
            depth=cfg.depth
        )
        model = IterativeViTClassifier(model=torch_model, config=cfg)
    elif cfg.model_type == 'vit':
        torch_model = VisionTransformer(
            emb_size=cfg.emb_size,
            depth=cfg.depth
        )
        model = ViTClassifier(model=torch_model, config=cfg)
    elif cfg.model_type == 'recursive_vit':
        torch_model = RecursiveViT(
            num_iterations=cfg.num_iterations,
            layer_idx=cfg.layer_idx,
            emb_size=cfg.emb_size,
            depth=cfg.depth
        )
        model = RecursiveViTClassifier(model=torch_model, config=cfg)
    else:
        raise ValueError(f"Invalid model type: {cfg.model_type}")

    # Setup logger
    logger = TensorBoardLogger(tensorboard_dir, name=cfg.experiment_name)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, f"version_{logger.version}"),
        filename=f"{cfg.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=True
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

# CUDA_VISIBLE_DEVICES=4,5 python main.py --experiment vit_iterative_2025-05-12-2352; 
# CUDA_VISIBLE_DEVICES=6,7 python main.py --experiment vit_iterative_2025-05-12-2356
# CUDA_VISIBLE_DEVICES=0,1 python main.py --experiment vit_base_2025-04-06-0134; 
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment vit_recursive_dev