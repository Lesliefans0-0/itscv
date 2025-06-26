import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.recursive_vit import RecursiveViT
from .iterative_vit_classifier import IterativeViTClassifier
from config import BaseConfig

class RecursiveViTClassifier(IterativeViTClassifier):
    def __init__(self, model: RecursiveViT, config: BaseConfig):
        super().__init__(model, config) 