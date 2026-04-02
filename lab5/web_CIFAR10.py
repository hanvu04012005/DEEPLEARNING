import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class CIFAR10_AdvancedCNN(nn.Module):
    def __init__(self):
        super(CIFAR10_AdvancedCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CIFAR10_AdvancedCNN().to(device)
model.load_state_dict(torch.load("cifar10_model.pth", map_location=device))
model.eval()

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

# ===== GRAD CAM =====
features = []
def hook_fn(module, input, output):
    features.append(output)

model.block3.register_forward_hook(hook_fn)

def grad_cam(image):
    features.clear()
    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    output = model(img_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    fmap = features[0].detach().cpu().numpy()[0]
    cam = np.mean(fmap, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32,32))
    cam = cam / (cam.max() + 1e-8)

    img_np = np.array(image.resize((32,32))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0

    overlay = heatmap * 0.6 + img_np * 0.4
    return (overlay * 255).astype(np.uint8)

# ===== PREDICT =====
def predict(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    # sort top 5
    idx = np.argsort(probs)[::-1][:5]

    table = []
    for i in idx:
        table.append([classes[i], f"{probs[i]*100:.2f}%"])

    cam_img = grad_cam(image)
    return table, cam_img

# ===== CSS (REAL PRODUCT STYLE) =====
css = """
body {
    background: #f4f6f8;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto;
    font-family: Inter, sans-serif;
}

/* header */
.header {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 4px;
}

.sub {
    color: #6b7280;
    margin-bottom: 20px;
}

/* card */
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
}

/* upload */
[data-testid="image"] {
    background: #fafafa !important;
    border: 1px dashed #d1d5db !important;
}

/* button */
button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

button:hover {
    background: #1d4ed8;
}
"""

# ===== UI =====
with gr.Blocks(css=css) as demo:

    gr.Markdown("<div class='header'>CIFAR-10 Vision AI</div>")
    gr.Markdown("<div class='sub'>Image Classification System</div>")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("**Input Image**")
                image_input = gr.Image(type="pil", label=None)
                btn = gr.Button("Run Inference")

        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("**Prediction Results**")
                table = gr.Dataframe(headers=["Class", "Confidence"])

            with gr.Group(elem_classes="card"):
                gr.Markdown("**Model Attention (Grad-CAM)**")
                cam_output = gr.Image()

    btn.click(fn=predict, inputs=image_input, outputs=[table, cam_output])

demo.launch()