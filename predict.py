import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# -------------------- 1. 加载模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("cat_dog_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------- 2. 图片预处理 --------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 3. 预测函数 --------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ["猫 🐱", "狗 🐶"]
    return classes[predicted.item()]

# -------------------- 4. 运行预测 --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict.py <图片路径>")
    else:
        result = predict(sys.argv[1])
        print(f"预测结果: {result}")