
import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

# --- 1. Load Mappings ---
# These files must be uploaded to the HF Space with this app.py
with open("index_to_class_id.json") as f:
    index_to_class_id = json.load(f)

with open("class_id_to_name.json") as f:
    class_id_to_name = json.load(f)

# --- 2. Load the Model ---
# We use pretrained=False (or weights=None) because we are loading our own trained weights
model = models.resnet18(pretrained=False) # Use pretrained=False for older torchvision
# Or for modern torchvision:
# model = models.resnet18(weights=None) 

# Modify the final layer to match our 200 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)

# Load the saved state dictionary
# We use map_location="cpu" so it runs on the Space's CPU
model_path = "resnet18_tiny_imagenet.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval() # Set model to evaluation mode

# --- 3. Define Image Transforms ---
# These MUST be the same as the validation transforms from your training script
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 4. Define Prediction Function ---
def predict(img_pil):
    """
    Takes a PIL Image, transforms it, and returns a dictionary
    of class probabilities.
    """
    # Apply transforms
    img_tensor = val_transform(img_pil).unsqueeze(0) # Add batch dimension

    # Get model predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        
    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Create a dictionary of {class_name: probability}
    confidences = {}
    for i in range(len(probabilities)):
        class_id = index_to_class_id[i]
        class_name = class_id_to_name.get(class_id, "Unknown")
        confidences[class_name] = float(probabilities[i])
        
    return confidences

# --- 5. Create Gradio Interface ---
# Get a few example images from your val set
example_paths = ["val_0.JPEG", "val_1.JPEG", "val_10.JPEG"]

title = "Tiny ImageNet Classifier (ResNet18)"
description = (
    "A ResNet18 model trained on the Tiny ImageNet (200 classes) dataset. "
    "Upload an image to see the model's top 5 predictions."
)

gr_interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title=title,
    description=description,
    examples=example_paths,
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    gr_interface.launch()
