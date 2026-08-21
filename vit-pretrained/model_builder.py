import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class ViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits
    
def create_vit(num_classes=3, device="cpu"):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # freeze only the vit backbone
    for param in model.vit.parameters():
        param.requires_grad = False

    # replace classifier head — new layer, requires_grad=True by default
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")

    return ViTWrapper(model).to(device)

if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=create_vit(num_classes=3,device=device)

    img = torch.randn(1, 3, 224, 224).to(device)
    out = model(img)
    print(f"Output shape: {out.shape}")