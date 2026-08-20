import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data_setup import download_data
from engine import train
from vit import ViT

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

train_dir,test_dir=download_data()
transform=transforms.Compose({
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
})
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset  = ImageFolder(test_dir,  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {train_dataset.classes}")

model=ViT(num_classes=NUM_CLASSES.to(DEVICE))

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)

results=train(model,train_loader,test_loader,loss_fn,optimizer,EPOCHS,DEVICE)

state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
torch.save(model.state_dict(),"vit_pizza_streak_sushi.pth")
print("Model Saved.")