"""
single_inference.py — Single-Image Wound Staging Inference
===========================================================
Paper: "Intelligent staging and adaptive intervention of chronic infected
       wounds via a deep-learning-assisted wearable sensing-to-therapeutic patch"
Journal: Nature Communications

Performs inference on a single wound image using a trained ResNet-18 model.
Generates a side-by-side visualization showing:
  - Left: Original image with predicted class and confidence score
  - Right: Grad-CAM heatmap highlighting diagnostically relevant regions

Usage:
  1. Set TARGET_IMG_PATH to your wound image
  2. Set MODEL_PATH to a trained checkpoint (e.g., best_model_fold5.pth)
  3. Run: python single_inference.py

Author: Qilun Yin
Date: 2025
"""
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms, models

# =========================================================================
# 1. 用户配置区
# =========================================================================
# 待测试的单张图片路径
TARGET_IMG_PATH = "/home/maoyuxin/code/linear_attn/classfiy/data/train/class0/S1.jpg"

# 模型权重路径 (请换成你效果最好的那个模型，通常是 fold5 或 fold1)
MODEL_PATH = "best_model_fold5.pth" 

# 类别定义
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
NUM_CLASSES = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. 基础设置
# =========================================================================
def set_nature_style():
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
set_nature_style()

# 数据预处理 (与训练保持一致)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================================
# 3. Grad-CAM 工具类
# =========================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # 注册 Hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 2. Zero grads
        self.model.zero_grad()
        
        # 3. Backward
        score = output[0, class_idx]
        score.backward()
        
        # 4. Generate CAM
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # GAP on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU & Normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        # 返回 CAM, 预测类别, 置信度
        return cam, class_idx, torch.softmax(output, dim=1)[0, class_idx].item()

# =========================================================================
# 4. 核心处理函数
# =========================================================================
def analyze_single_image(img_path, model_path):
    # --- A. 尝试从路径推断 GT ---
    # 假设路径是 .../class0/xxx.jpg，我们取父目录名
    folder_name = os.path.basename(os.path.dirname(img_path))
    gt_label = "Unknown"
    # 简单的匹配逻辑，你可以根据实际文件夹名修改
    if "class0" in folder_name.lower() or "type0" in folder_name.lower(): gt_idx = 0
    elif "class1" in folder_name.lower() or "type1" in folder_name.lower(): gt_idx = 1
    elif "class2" in folder_name.lower() or "type2" in folder_name.lower(): gt_idx = 2
    elif "class3" in folder_name.lower() or "type3" in folder_name.lower(): gt_idx = 3
    else: gt_idx = -1
    
    if gt_idx != -1:
        gt_label = CLASS_NAMES[gt_idx]
    
    print(f"Processing: {os.path.basename(img_path)}")
    print(f"Inferred GT: {gt_label}")

    # --- B. 加载模型 ---
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model.to(DEVICE)
    model.eval()

    # --- C. 图像处理 ---
    # 1. 读取原始图片用于显示
    raw_img = Image.open(img_path).convert('RGB')
    raw_img_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
    raw_img_np = np.array(raw_img_resized) / 255.0 # [0,1]
    
    # 2. 预处理用于模型输入
    input_tensor = preprocess(raw_img).unsqueeze(0).to(DEVICE)

    # --- D. Grad-CAM 推理 ---
    # 锁定 Layer4 (最后一层卷积)
    grad_cam = GradCAM(model, model.layer4[-1])
    
    # 获取热力图
    # 注意：这里我们让 Grad-CAM 计算“预测类别”的热力图
    mask, pred_idx, conf = grad_cam(input_tensor)
    
    pred_label = CLASS_NAMES[pred_idx]

    # --- E. 可视化绘图 ---
    plt.figure(figsize=(10, 5))

    # 子图 1: 原始图片 + 预测信息
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(raw_img_resized)
    
    # 标题逻辑：绿色表示对，红色表示错
    if gt_idx != -1:
        is_correct = (pred_idx == gt_idx)
        color = '#006400' if is_correct else '#CC0000' # Green / Red
        title_text = f"Ground Truth: {gt_label}\nPrediction: {pred_label}\nConfidence: {conf:.2%}"
    else:
        color = 'black'
        title_text = f"Prediction: {pred_label}\nConfidence: {conf:.2%}"
        
    ax1.set_title(title_text, color=color, fontweight='bold', fontsize=12, pad=10)
    ax1.axis('off')

    # 子图 2: Grad-CAM 热力图叠加
    ax2 = plt.subplot(1, 2, 2)
    
    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR -> RGB
    
    # 叠加 (60% 原图 + 40% 热力图)
    overlay = 0.6 * raw_img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    ax2.imshow(overlay)
    ax2.set_title("Grad-CAM Activation\n(Where the model looked)", fontweight='bold', fontsize=12, pad=10)
    ax2.axis('off')

    plt.tight_layout()
    
    # 保存结果
    save_name = f"result_{os.path.basename(img_path).split('.')[0]}.pdf"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"Result saved to {save_name}")
    plt.show()

# =========================================================================
# 5. 执行
# =========================================================================
if __name__ == "__main__":
    if os.path.exists(TARGET_IMG_PATH):
        analyze_single_image(TARGET_IMG_PATH, MODEL_PATH)
    else:
        print(f"Error: Image not found at {TARGET_IMG_PATH}")