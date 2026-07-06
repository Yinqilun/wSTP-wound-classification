"""
inference_ensemble.py — Batch Evaluation & Advanced Visualization
=================================================================
Paper: "Intelligent staging and adaptive intervention of chronic infected
       wounds via a deep-learning-assisted wearable sensing-to-therapeutic patch"
Journal: Nature Communications

Comprehensive evaluation pipeline for the wound staging model, generating:
  1. t-SNE feature embedding visualization (high-contrast, colorblind-safe)
  2. Grad-CAM heatmaps for model interpretability
  3. Confidence distribution histograms (correct vs. incorrect predictions)

All outputs are saved as publication-ready vector PDFs (300 DPI, Nature style).

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
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import torch.nn.functional as F

# =========================================================================
# 1. 配置参数
# =========================================================================
# 指向验证集/测试集
DATA_ROOT = "/home/maoyuxin/code/linear_attn/classfiy/data/train_split/train"
# 指向其中一个训练好的模型权重 (例如 Fold 5)
MODEL_PATH = "best_model_fold5.pth" 

BATCH_SIZE = 32
NUM_CLASSES = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

print(f"Device: {DEVICE}")

# =========================================================================
# 2. 风格设置
# =========================================================================
def set_nature_style():
    sns.set_context("paper", font_scale=1.4)
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

# =========================================================================
# 3. 数据与模型准备
# =========================================================================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_data_and_model():
    # Data
    dataset = datasets.ImageFolder(root=DATA_ROOT, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # t-SNE 不需要 shuffle
    
    # Model
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    # 加载权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model: {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model.to(DEVICE)
    model.eval()
    return dataset, loader, model

# =========================================================================
# 4. 高级可视化一：t-SNE 特征分布图
# =========================================================================
def plot_tsne(model, loader, classes):
    print("Generating t-SNE plot (extracting features)...")
    
    # 1. Hook 提取特征 (保持不变)
    features_list = []
    labels_list = []
    
    def hook_fn(module, input, output):
        features_list.append(output.flatten(1).cpu().detach().numpy())

    handle = model.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            _ = model(x)
            labels_list.extend(y.numpy())
            
    handle.remove()
    
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    
    # 2. t-SNE 降维 (保持不变)
    print("Running t-SNE algorithm...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)
    
    # 3. 绘图 (【重点修改部分】)
    plt.figure(figsize=(10, 8))
    
    # === A. 自定义高对比度颜色 (Colorblind Safe) ===
    # 这种配色方案在色彩心理学上区分度最高
    # 0: 蓝色 (Blue), 1: 橙色 (Orange), 2: 绿色 (Green), 3: 红色 (Red)
    # 如果你有更多类别，可以加紫色('#9467bd') 和 棕色('#8c564b')
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # === B. 自定义形状 (Markers) ===
    # 圆圈(o), 方块(s), 三角(^), 菱形(D)
    # 这样即使是黑白打印也能分清
    custom_markers = ['o', 's', '^', 'D'] 
    
    for i, class_name in enumerate(classes):
        idxs = labels == i
        
        # 确保颜色和形状循环使用（防止类别数超过4个时报错）
        color = custom_colors[i % len(custom_colors)]
        marker = custom_markers[i % len(custom_markers)]
        
        plt.scatter(
            X_embedded[idxs, 0], X_embedded[idxs, 1], 
            label=class_name, 
            c=color,          # 颜色
            marker=marker,    # 形状
            alpha=0.8,        # 透明度稍微调高一点，看着更清楚
            s=60,             # 点的大小稍微调大
            edgecolors='k',   # 给每个点加个黑色边框，对比度拉满
            linewidth=0.5
        )
        
    # 4. 装饰
    plt.title('t-SNE Feature Visualization', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('Dimension 2', fontsize=14, fontweight='bold')
    
    # 图例放在图外，避免遮挡数据
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, 
               title="Categories", fontsize=12, title_fontsize=14, frameon=False)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = 'advanced_viz_tsne_high_contrast.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Saved optimized t-SNE to: {save_path}")

# =========================================================================
# 5. 高级可视化二：Grad-CAM (可解释性热力图)
# =========================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册 Hook
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass (Target class score)
        score = output[0, class_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients.data.cpu().numpy()[0] # [512, 7, 7]
        activations = self.activations.data.cpu().numpy()[0] # [512, 7, 7]
        
        # Global Average Pooling of gradients (Weights)
        weights = np.mean(gradients, axis=(1, 2)) # [512]
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize to 0-1
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, class_idx, torch.softmax(output, dim=1)[0, class_idx].item()

def visualize_gradcam(dataset, model, classes, num_samples=5):
    print("Generating Grad-CAM Heatmaps...")
    
    # 锁定 ResNet18 的最后一层卷积层 (Layer 4)
    # 这一层包含了最高级的语义特征
    grad_cam = GradCAM(model, model.layer4[-1])
    
    # 随机取样
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(12, 4 * num_samples))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
        
        # 获取 CAM
        mask, pred_idx, conf = grad_cam(img_tensor)
        
        # 处理原始图片用于显示
        img_np = img_tensor.cpu().squeeze().numpy().transpose((1, 2, 0))
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # 创建热力图 overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1] # BGR to RGB
        
        # 叠加: 60% 原图 + 40% 热力图
        cam_result = 0.6 * img_np + 0.4 * heatmap
        cam_result = np.clip(cam_result, 0, 1)
        
        # --- 绘图 (左：原图，右：Grad-CAM) ---
        # 1. Original
        ax1 = plt.subplot(num_samples, 2, 2*i + 1)
        ax1.imshow(img_np)
        ax1.set_title(f"True: {classes[label]}", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Grad-CAM
        ax2 = plt.subplot(num_samples, 2, 2*i + 2)
        ax2.imshow(cam_result)
        color = 'green' if label == pred_idx else 'red'
        ax2.set_title(f"Pred: {classes[pred_idx]} ({conf:.1%})\nTarget: {classes[label]}", 
                      color=color, fontsize=12, fontweight='bold')
        ax2.axis('off')
        
    plt.tight_layout()
    plt.savefig('advanced_viz_gradcam.pdf')
    plt.show()
    print("Saved: advanced_viz_gradcam.pdf")

# =========================================================================
# 6. 高级可视化三：置信度直方图 (Confidence Histogram)
# =========================================================================
def plot_confidence_hist(model, loader):
    print("Generating Confidence Histogram...")
    
    correct_confs = []
    wrong_confs = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            
            # 获取预测的类别和对应的置信度
            max_probs, preds = torch.max(probs, dim=1)
            
            for i in range(len(y)):
                if preds[i] == y[i]:
                    correct_confs.append(max_probs[i].item())
                else:
                    wrong_confs.append(max_probs[i].item())
                    
    plt.figure(figsize=(10, 6))
    
    # 绘制两个直方图
    sns.histplot(correct_confs, color="green", label="Correct Predictions", kde=True, bins=20, alpha=0.5, element="step")
    sns.histplot(wrong_confs, color="red", label="Wrong Predictions", kde=True, bins=20, alpha=0.5, element="step")
    
    plt.xlabel("Model Confidence (Probability)", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title("Prediction Confidence Distribution", fontsize=14, pad=15)
    plt.legend()
    plt.xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('advanced_viz_confidence.pdf')
    plt.show()
    print("Saved: advanced_viz_confidence.pdf")

# =========================================================================
# 7. 主程序
# =========================================================================
if __name__ == "__main__":
    # 1. 加载
    dataset, loader, model = load_data_and_model()
    
    # 2. t-SNE (查看特征可分性)
    plot_tsne(model, loader, CLASS_NAMES)
    
    # 3. Grad-CAM (查看模型关注点)
    # 注意：Grad-CAM 需要反向传播，所以不能用 eval 模式的完全无梯度环境
    # 我们上面的 GradCAM 类里手动处理了 zero_grad
    visualize_gradcam(dataset, model, CLASS_NAMES, num_samples=6)
    
    # 4. 置信度分布 (查看模型是否盲目自信)
    plot_confidence_hist(model, loader)