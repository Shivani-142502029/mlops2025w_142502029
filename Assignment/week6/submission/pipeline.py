import ssl 
ssl._create_default_https_context = ssl._create_unverified_context 
import numpy as np 
import os 
import json 
import tomli 
import torch 
from torchvision import transforms 
from PIL import Image 
import torchvision.models as models 
from torchvision.models import ( ResNet34_Weights, 
                                ResNet50_Weights, ResNet101_Weights, ResNet152_Weights ) 
# ---------- Step 1: Load Configs ---------- 
with open("config.json", "r") as f: 
    config = json.load(f) 
with open("params.toml", "rb") as f: 
    params = tomli.load(f) 
with open("grid.json", "r") as f: 
    grid = json.load(f) 
model_dict = { "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1), 
              "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1), 
              "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V1), 
              "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V1), } 
preprocess = transforms.Compose([ 
    transforms.Resize(256), # resize short side to 256 
    transforms.CenterCrop(224), # crop to 224x224 
    transforms.ToTensor(), # convert to tensor 
    transforms.Normalize( # normalize as ResNet expects 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225] 
    ), 
]) 
image_folder = config["data_source"] 
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
               if f.lower().endswith((".png", ".jpg", ".jpeg"))] 
for model_name in config["model"]: 
    data_source = config["data_source"] 
    print(f"Using {model_name} on data from {data_source}") 


    # ---------- Step 2: Load Pretrained ResNet ---------- 
    model_fn, weight_cls = model_dict[model_name] 
    model = model_fn(weights=weight_cls) 
    model.eval() 


    # ---------- Step 3: Load Hyperparameters ---------- 
    lr = params[model_name]["learning_rate"] 
    batch_size = params[model_name]["batch_size"] 
    print(f"Default hyperparams → lr: {lr}, batch_size: {batch_size}") 

    # ---------- Step 4: Real Inference ----------
    results = []
    for lr in grid["learning_rates"]:
        for opt in grid["optimizers"]:
            for mom in grid["momentum"]:
                for img_path in image_files:
                    img = Image.open(img_path).convert("RGB")
                    input_tensor = preprocess(img).unsqueeze(0)  # add batch dimension

                    with torch.no_grad():
                        output = model(input_tensor)

                    # Convert tensor to numpy for saving
                    output_np = output.numpy().tolist()

                    print(f"Image: {os.path.basename(img_path)}, Output shape: {output.shape}")
                    print(f"Output values (first 5): {output_np[0][:5]}")  # print first 5 numbers

                    config_str ={
                        "model": model_name,
                        "image": os.path.basename(img_path),
                        "lr": lr,
                        "optimizer": opt,
                        "momentum": mom,
                        "output": output_np[0][:5]  # full 1000 values
                    }

                    # config_str = f"lr={lr}, optimizer={opt}, momentum={mom}"
                    print("Grid search config:", config_str)
                    results.append({"grid_search": config_str})

    with open("results.txt", "a") as f:
        f.write(f"\n=== Results for {model_name} ===\n")
        for r in results:
            f.write(json.dumps(r) + "\n")  # save as JSON lines

print("The results have been saved in results.txt file")
