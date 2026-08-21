import torch
def train_step(model,loader,loss_fn,optimizer,device):
    model.train()
    total_loss=0
    total_accuracy=0
    for X,y in loader:
        X,y=X.to(device),y.to(device)
        optimizer.zero_grad()
        logits=model(X)
        loss=loss_fn(logits,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        total_accuracy+=(logits.argmax(dim=1)==y).float().mean().item()
    return total_loss/len(loader),total_accuracy/len(loader)

def test_step(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == y).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)

def train(model, train_loader, test_loader, loss_fn, optimizer, epochs, device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss,train_acc=train_step(model,train_loader,loss_fn,optimizer,device)
        test_loss,test_acc=test_step(model,test_loader,loss_fn,device)


        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results['test_acc'].append(test_acc)
        results['test_acc'].append(test_acc)