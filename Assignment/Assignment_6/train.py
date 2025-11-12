import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm  # For a nice progress bar
import wandb
import os

# Import our custom data loader
from dataset import get_data_loaders

# --- 1. Configuration ---
# All hyperparameters are in one place
config = {
    "project_name": "mlops-assignment6-resnet",
    "run_name": "baseline-resnet18",
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 128, # You might need to lower this if you get CUDNN errors (e.g., to 64 or 32)
    "model_architecture": "resnet18",
    "dataset": "Tiny-ImageNet-200",
    "data_path": "./tiny-imagenet-200" # Path to the unzipped folder
}

# --- 2. W&B Initialization ---
# (You'll need to run 'wandb login' in your terminal first)
print("Logging into W&B...")
wandb.login()

run = wandb.init(
    project=config["project_name"],
    name=config["run_name"],
    config=config
)

print("W&B Run initialized.")

# --- 3. Data Loading ---
print("Loading data...")
train_loader, val_loader, class_names = get_data_loaders(
    config["data_path"], 
    config["batch_size"]
)
num_classes = len(class_names)
print(f"Data loaded. Found {num_classes} classes.")

# --- 4. Model Setup ---
print(f"Loading model: {config['model_architecture']}...")
# Load a pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for our 200 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to {device}.")

# --- 5. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# --- 6. W&B Watch ---
# This will log gradients and model parameters
wandb.watch(model, log="all", log_freq=100)

# --- 7. Training & Validation Loop ---
print("Starting training...")
for epoch in range(config["epochs"]):
    # --- Training Phase ---
    model.train() # Set model to training mode
    train_loss = 0.0
    
    # Use tqdm for a progress bar
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # No gradients needed for validation
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # --- 8. Logging Metrics to W&B ---
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1} Summary: "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Accuracy: {accuracy:.2f}%")
    
    # Log metrics to W&B dashboard
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "accuracy": accuracy
    })

# --- 9. Save Model as W&B Artifact ---
print("Training finished. Saving model artifact...")
model_path = "resnet18_tiny_imagenet.pth"
torch.save(model.state_dict(), model_path)

# Create a W&B Artifact
artifact = wandb.Artifact(
    name=f"{config['model_architecture']}-final", 
    type="model",
    description="Final trained model state dictionary for Tiny ImageNet"
)
artifact.add_file(model_path)

# Log the artifact to W&B
run.log_artifact(artifact)
print(f"Model artifact saved as {artifact.name}")

# Finish the W&B run
run.finish()
print("W&B run finished.")