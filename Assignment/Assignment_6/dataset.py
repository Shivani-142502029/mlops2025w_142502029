import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import os

def get_data_loaders(data_dir, batch_size, num_workers=4):
    """
    Creates and returns the Tiny ImageNet DataLoaders.
    """
    
    # --- 1. Define Image Transforms ---
    # Transforms are operations applied to each image (resize, normalize, etc.)
    # ResNet models expect 224x224 images
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # --- 2. Create Training Dataset ---
    # The 'train' folder is structured perfectly for ImageFolder
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = ImageFolder(train_dir, transform=train_transform)

    # --- 3. Create Validation Dataset (Custom Class) ---
    val_dir = os.path.join(data_dir, 'val')
    val_dataset = TinyImageNetValDataset(val_dir, train_dataset.class_to_idx, transform=val_transform)

    # --- 4. Create DataLoaders ---
    # DataLoaders batch the data and shuffle it
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

class TinyImageNetValDataset(Dataset):
    """
    Custom PyTorch Dataset for the Tiny ImageNet validation set.
    """
    def __init__(self, val_dir, class_to_idx, transform=None):
        self.val_dir = os.path.join(val_dir, 'images')
        self.transform = transform
        self.class_to_idx = class_to_idx
        
        self.image_paths = []
        self.labels = []
        
        # Load validation annotations
        annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                    
                img_name, class_id = parts[0], parts[1]
                
                # Full path to the image
                self.image_paths.append(os.path.join(self.val_dir, img_name))
                
                # Get the integer label (index) from the class ID
                self.labels.append(self.class_to_idx[class_id])

    def __len__(self):
        # Returns the total number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Loads and returns a sample (image, label) at the given index
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image using PIL
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label