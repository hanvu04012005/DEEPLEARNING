import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

# ===== DEFINE MODEL LẠI =====
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ===== LOAD MODEL =====
device = torch.device("cpu")

checkpoint = torch.load("plant_model.pth", map_location=device)

model = CNN(checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

class_names = checkpoint["class_names"]


# ===== TRANSFORM (GIỐNG TRAIN) =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ===== PREDICT FUNCTION =====
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    # trả top 3
    result = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    return result


# ===== GRADIO UI =====
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🌱 Plant Disease Detection",
    description="Upload ảnh lá cây để nhận diện bệnh"
)

demo.launch()