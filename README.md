# 猫狗图像分类器

基于 PyTorch 的图像分类项目，用于区分猫和狗。项目对比了两种方案：迁移学习（ResNet18）与从零训练（自定义 CNN），并提供 FastAPI Web 界面供交互测试。

## 项目结构

- `data/train/`：训练数据集（Kaggle Dogs vs Cats，25000张图片）
- `data/test/`：测试图片
- `train_resnet.py`：ResNet18 迁移学习训练脚本
- `train_cnn.py`：自定义 CNN 训练脚本
- `predict.py`：ResNet18 模型预测脚本
- `evaluate_cnn.py`：自定义 CNN 模型评估脚本
- `app.py`：FastAPI Web 服务
- `templates/index.html`：Web 前端页面

## 环境依赖

- Python 3.12
- PyTorch 2.8.0
- torchvision
- Pillow
- scikit-learn
- fastapi
- uvicorn

## 如何运行

### 1. 激活环境

conda activate torch_gpu

### 2. 训练模型

python train_resnet.py
python train_cnn.py

### 3. 命令行预测

python predict.py "图片路径"

### 4. 启动 Web 界面

python -m uvicorn app:app --reload

浏览器打开 http://127.0.0.1:8000 ，上传图片即可查看预测结果。

## 训练结果

### ResNet18（迁移学习）

使用 ImageNet 预训练权重，训练 5 个 epoch。

| Epoch | 训练准确率 | 验证准确率 |
|-------|----------|----------|
| 1     | 91.81%   | 93.00%   |
| 2     | 94.78%   | 94.26%   |
| 3     | 95.64%   | 94.30%   |
| 4     | 96.03%   | 91.76%   |
| 5     | 96.56%   | **94.86%** |

**最终验证集准确率：94.86%**

### 自定义 CNN（从零训练）

不使用预训练权重，从零开始训练 5 个 epoch。

| Epoch | 训练准确率 | 验证准确率 |
|-------|----------|----------|
| 1     | 60.48%   | 65.68%   |
| 2     | 67.75%   | 72.10%   |
| 3     | 71.30%   | 74.16%   |
| 4     | 73.29%   | 76.18%   |
| 5     | 75.19%   | **78.24%** |

**最终验证集准确率：78.24%**

## 模型对比

| 模型 | 验证准确率 | 训练方式 |
|------|-----------|---------|
| ResNet18 | **94.86%** | 迁移学习（ImageNet 预训练） |
| 自定义 CNN | 78.24% | 从零训练 |

**结论**：在相同数据和训练轮数下，迁移学习显著优于从零训练。

## 自定义 CNN 的问题分析

对自定义 CNN 进行混淆矩阵和分类报告分析：

分类报告（自定义 CNN）：
              precision    recall  f1-score   support
           猫       0.74      0.90      0.81      2513
           狗       0.87      0.68      0.77      2487
    accuracy                           0.79      5000

**问题**：模型对猫的召回率为 90%，但对狗的召回率仅 68%，存在明显偏科。

**原因**：
1. 从零训练，5 个 epoch 不足以让模型学到足够的判别性特征
2. 模型参数量较小，表达能力有限

**改进方向**：增加训练轮数、引入更强的数据增强、增加网络深度。

## Web 界面

使用 FastAPI 搭建，支持浏览器上传图片并实时返回预测结果与置信度。

启动方式：

python -m uvicorn app:app --reload

访问 http://127.0.0.1:8000

## 后续改进方向

- 增加数据增强策略，缓解过拟合
- 尝试更大的模型（如 ResNet50）
- 对自定义 CNN 增加训练轮数
- 将 Web 服务部署到云端

## 预测示例

python predict.py "data/test/1.jpg"
# 输出：预测结果: 狗 🐶

python predict.py "data/test/5.jpg"
# 输出：预测结果: 猫 🐱"