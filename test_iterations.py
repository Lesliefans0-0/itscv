import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import argparse
import os
from pytorch_lightning.loggers import TensorBoardLogger
from models.iterative_vit import IterativeViT
from lightning_models.iterative_vit_classifier import IterativeViTClassifier
from data.imagenet_datamodule import CustomImageNetDataModule
from config import get_experiment_config

def test_model_with_iterations(checkpoint_path, experiment_name, max_iterations=10):
    # Load the experiment config
    cfg = get_experiment_config(experiment_name)
    
    # Setup TensorBoard logger
    experiment_dir = os.path.dirname(checkpoint_path)
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    logger = TensorBoardLogger(tensorboard_dir, name=f"{experiment_name}_test")
    
    # Initialize data module
    data_module = CustomImageNetDataModule(
        imagenet_dir=cfg.imagenet_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=logger
    )
    
    # Test different numbers of iterations
    iteration_accs = []
    
    for num_iter in range(1, max_iterations + 1):
        # Create model with current number of iterations
        torch_model = IterativeViT(
            num_iterative_tokens=cfg.num_iterative_tokens,
            num_iterations=num_iter,
            layer_idx=cfg.layer_idx,
            emb_size=cfg.emb_size
        )
        model = IterativeViTClassifier(model=torch_model, lr=float(cfg.learning_rate))
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Test the model
        results = trainer.test(model, data_module)
        final_acc = results[0]['test_acc']
        iteration_accs.append(final_acc)
        
        # Log to TensorBoard
        logger.experiment.add_scalar('test/accuracy', final_acc, num_iter)
        
        print(f"Number of iterations: {num_iter}, Test Accuracy: {final_acc:.4f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iterations + 1), iteration_accs, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Number of Iterations')
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(experiment_dir, 'iteration_accuracy_curve.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment config')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum number of iterations to test')
    args = parser.parse_args()
    
    test_model_with_iterations(args.checkpoint, args.experiment, args.max_iterations) 

# CUDA_VISIBLE_DEVICES=2,5 python test_iterations.py --checkpoint "experiments/vit_iterative_2025-04-06-0134/checkpoints/last.ckpt" --experiment vit_iterative_2025-04-06-0134 --max_iterations 10