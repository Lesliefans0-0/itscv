import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
import PIL.Image as PImage
import os

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, imagenet_dir, batch_size=128, num_workers=8):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageNet(
                self.imagenet_dir, split='train',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            )
            
            self.val_dataset = datasets.ImageNet(
                self.imagenet_dir, split='val',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        # Note: ImageNet doesn't release its test set publicly
        # Following standard practice, we use the validation set for testing
        return self.val_dataloader()

def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img

class CustomImageNetDataModule(ImageNetDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetFolder(
                root=os.path.join(self.imagenet_dir, 'train'), 
                loader=pil_loader, 
                extensions=IMG_EXTENSIONS, 
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            )
            self.val_dataset = DatasetFolder(
                root=os.path.join(self.imagenet_dir, 'val'), 
                loader=pil_loader, 
                extensions=IMG_EXTENSIONS, 
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            )
