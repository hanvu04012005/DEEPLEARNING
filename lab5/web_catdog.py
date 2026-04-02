import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

# ===== DEFINE MODEL =====
class CatDog_CNN_Advanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ===== LOAD MODEL =====
device = torch.device("cpu")

checkpoint = torch.load("catdog_model.pth", map_location=device)

model = CatDog_CNN_Advanced()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

class_names = checkpoint["class_names"]

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((checkpoint["input_size"], checkpoint["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        checkpoint["normalize_mean"],
        checkpoint["normalize_std"]
    )
])


# ===== PREDICT =====
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    return {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }


# ===== GRADIO UI =====
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🐶🐱 Cat vs Dog Classifier",
    description="Upload ảnh để phân biệt mèo hay chó"
)

demo.launch()