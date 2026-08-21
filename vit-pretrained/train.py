import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data_setup import download_data
from model_builder import create_vit
from engine import train
from transformers import AutoImageProcessor
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

train_dir,test_dir=download_data()

processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean,
                         std=processor.image_std)
])

train_dataset=ImageFolder(train_dir,transform=transform)
test_dataset=ImageFolder(test_dir,transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes:{train_dataset.classes}")

model=create_vit(num_classes=NUM_CLASSES,device=DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
    
loss_fn=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

results=train(model,train_loader,test_loader,loss_fn,optimizer,EPOCHS,DEVICE)

torch.save(model.state_dict(), "vit_pretrained_pizza_steak_sushi.pth")
print("Model saved.")