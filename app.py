"""猫狗分类 FastAPI Web 界面。

模型结构与 train_resnet.py / predict.py 保持一致：
ResNet18 + 二分类全连接层，输入 128x128，输出 [猫, 狗]。
启动方式：uvicorn app:app --reload
"""

import io
import os

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# -------------------- 1. 基础配置 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "cat_dog_resnet18.pth"
MODEL_PATH = MODEL_NAME if os.path.exists(MODEL_NAME) else os.path.join(BASE_DIR, MODEL_NAME)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

CLASS_NAMES = ["猫", "狗"]
CLASS_EMOJI = ["🐱", "🐶"]
CLASS_KEYS = ["cat", "dog"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 2. 加载模型 --------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -------------------- 3. 图片预处理 --------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------- 4. 预测函数 --------------------
def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    index = predicted.item()
    return {
        "index": index,
        "label": CLASS_NAMES[index],
        "emoji": CLASS_EMOJI[index],
        "key": CLASS_KEYS[index],
        "confidence": round(probs[index].item() * 100, 2),
        "cat_prob": round(probs[0].item() * 100, 2),
        "dog_prob": round(probs[1].item() * 100, 2),
    }


def _wants_json(request: Request) -> bool:
    """浏览器表单请求返回 HTML 页面，客户端带 Accept: application/json 时返回 JSON。"""
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


def _render(request: Request, result=None, error=None, filename=None, status_code=200):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result, "error": error, "filename": filename},
        status_code=status_code,
    )


# -------------------- 5. FastAPI 应用 --------------------
app = FastAPI(title="猫狗分类器", description="ResNet18 猫狗二分类 Web 界面")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render(request)


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or "upload"

    if not content:
        if _wants_json(request):
            return JSONResponse({"success": False, "error": "上传的文件为空"}, status_code=400)
        return _render(request, error="上传的文件为空", status_code=400)

    try:
        result = predict_image(content)
    except (UnidentifiedImageError, OSError):
        if _wants_json(request):
            return JSONResponse({"success": False, "error": "无法识别的图片格式"}, status_code=400)
        return _render(request, error="无法识别的图片格式，请上传 JPG / PNG 等常见图片", status_code=400)

    if _wants_json(request):
        return JSONResponse({
            "success": True,
            "filename": filename,
            "label": result["label"],
            "class": result["key"],
            "confidence": result["confidence"],
            "cat_prob": result["cat_prob"],
            "dog_prob": result["dog_prob"],
        })

    return _render(request, result=result, filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
