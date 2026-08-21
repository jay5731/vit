import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data_setup import download_data
from model_builder import create_vit
from engine import train

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

test_dir,test_dir=download_data()

processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean,
                         std=processor.image_std)
])