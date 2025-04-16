import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.recursive_vit import RecursiveViT
from .iterative_vit_classifier import IterativeViTClassifier

class RecursiveViTClassifier(IterativeViTClassifier):
    def __init__(self, model: RecursiveViT, lr=3e-4):
        super().__init__(model, lr) 