import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader
from torchvision.models import resnet18

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=str, default="cifar100-hf-demo")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()

def get_dataloaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100. * correct / total

def evaluate(model, device, loader, criterion):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            l = criterion(outputs, targets)
            loss += l.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return loss/total, 100.*correct/total

def main():
    args = parse_args()
    wandb.init(project=args.project, entity=args.entity, config=vars(args))
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainloader, testloader = get_dataloaders(cfg.batch_size)

    model = resnet18(num_classes=100)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, device, trainloader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, testloader, criterion)
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                   "test_loss": test_loss, "test_acc": test_acc})
        print(f"Epoch {epoch+1}: train_acc={train_acc:.2f} test_acc={test_acc:.2f}")
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/model.pt")
            # log artifact
            artifact = wandb.Artifact("resnet18-cifar100", type="model", metadata={"test_acc": best_acc})
            artifact.add_file("outputs/model.pt")
            wandb.log_artifact(artifact)
    print("Best test acc:", best_acc)

if __name__ == "__main__":
    main()
