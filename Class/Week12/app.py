import os, time, io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import gradio as gr

MODEL_PATH = "outputs/model.pt"

# If model not present, try download via W&B (requires WANDB_API_KEY secret in Space or env)
if not os.path.exists(MODEL_PATH):
    try:
        import wandb
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            api = wandb.Api()
            artifact = api.artifact(os.environ.get("WANDB_ARTIFACT", "YOUR_WANDB_ENTITY/cifar100-hf-demo/resnet18-cifar100:latest"))
            artifact.download(root="outputs")
            print("Downloaded model via W&B artifact.")
        else:
            print("WANDB_API_KEY not set; cannot download artifact.")
    except Exception as e:
        print("Error downloading artifact via W&B:", e)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(num_classes=100)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673,0.2564,0.2762))
])

def predict_image(img):
    start = time.time()
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, idx = probs.max(1)
        class_idx = int(idx.item())
        conf_val = float(conf.item())
    latency = (time.time() - start) * 1000.0
    return {"class_idx": class_idx, "confidence": round(conf_val,4), "latency_ms": round(latency,2)}

iface = gr.Interface(fn=predict_image, inputs=gr.Image(type="pil"), outputs="json", title="CIFAR-100 demo")
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
