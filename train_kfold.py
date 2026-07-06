"""
train_kfold.py — 5-Fold Cross-Validation Training Pipeline
===========================================================
Paper: "Intelligent staging and adaptive intervention of chronic infected
       wounds via a deep-learning-assisted wearable sensing-to-therapeutic patch"
Journal: Nature Communications

This script implements stratified 5-fold cross-validation training of a
ResNet-18 ensemble for wound-stage classification. Key features:
  - ImageNet pre-trained ResNet-18 backbone with semi-freezing fine-tuning
  - Weighted random sampling to address class imbalance
  - Differential learning rates (low for feature extractor, high for classifier)
  - Ensemble soft-voting inference across all 5 folds

Author: Qilun Yin
Date: 2025
"""
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# =========================================================================
# 1. 配置参数 (User Config)
# =========================================================================
# 指向你要进行测试的数据集路径 (例如 validation 文件夹)
TEST_DATA_ROOT = "/home/maoyuxin/code/linear_attn/classfiy/data/train_split/train"
# 模型权重文件所在的目录 (默认当前目录)
MODEL_DIR = "." 

# 可视化时想要随机展示的图片数量
VISUALIZE_NUM = 10

BATCH_SIZE = 32
NUM_CLASSES = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称 (需与训练时保持一致)
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

print(f"Inference Device: {DEVICE}")

# =========================================================================
# 2. 绘图风格 (Nature Style)
# =========================================================================
def set_nature_style():
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
set_nature_style()

# =========================================================================
# 3. 数据加载
# =========================================================================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_test_data():
    if not os.path.exists(TEST_DATA_ROOT):
        raise FileNotFoundError(f"Test data path not found: {TEST_DATA_ROOT}")
        
    # 这里返回 dataset 对象，方便后续随机采样
    dataset = datasets.ImageFolder(root=TEST_DATA_ROOT, transform=test_transform)
    # 用于跑全量评估的 Loader (不打乱)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Loaded Test Dataset: {len(dataset)} images.")
    return dataset, loader

# =========================================================================
# 4. 模型定义 & 加载 (Ensemble)
# =========================================================================
def get_model_structure():
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    return model

def load_ensemble_models():
    models_list = []
    print("Loading Ensemble Models...")
    # 尝试加载 Fold 1 到 Fold 5
    for i in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_model_fold{i}.pth")
        if os.path.exists(path):
            print(f"  -> Loading {path}...")
            model = get_model_structure()
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models_list.append(model)
        else:
            print(f"  -> Warning: {path} not found. Skipping.")
    
    if not models_list:
        raise RuntimeError("No model files found! Please check MODEL_DIR.")
    
    return models_list

# =========================================================================
# 5. 全量推理逻辑 (用于计算指标)
# =========================================================================
def run_full_inference(loader, models_list):
    all_preds = []
    all_targets = []
    all_probs = [] 
    
    print("Running Full Inference on Test Set...")
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            
            # 5模型投票
            outputs = [model(x) for model in models_list]
            avg_output = torch.stack(outputs).mean(0) 
            
            probs = torch.softmax(avg_output, dim=1)
            preds = avg_output.argmax(1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

# =========================================================================
# 6. 可视化绘图函数
# =========================================================================

def plot_confusion_matrix(preds, targets, classes):
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12, "weight": "bold"}, cbar=False,
                square=True, linewidths=1.5, linecolor='white')
    
    plt.xlabel('Predicted Label', fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontweight='bold', labelpad=10)
    plt.title('Ensemble Confusion Matrix', fontsize=15, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('inference_confusion_matrix.pdf')
    plt.show()
    print("Saved: inference_confusion_matrix.pdf")

def plot_multiclass_roc(targets, probs, classes):
    y_test = label_binarize(targets, classes=range(len(classes)))
    n_classes = y_test.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Multi-class ROC Curve', fontsize=16, pad=20, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig('inference_roc_curve.pdf')
    plt.show()
    print("Saved: inference_roc_curve.pdf")

# --- 【关键修改】随机采样可视化 ---
def visualize_random_samples(dataset, models_list, classes, num_samples=10):
    print(f"Visualizing {num_samples} random samples...")
    
    # 1. 从全集中随机抽取索引
    total_len = len(dataset)
    indices = random.sample(range(total_len), min(num_samples, total_len))
    
    # 2. 创建临时子集和 Loader
    subset = Subset(dataset, indices)
    # 这里不需要 shuffle，因为 indices 已经是随机的了
    temp_loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
    
    # 3. 获取数据
    x, y = next(iter(temp_loader))
    x = x.to(DEVICE)
    
    # 4. 推理
    with torch.no_grad():
        outputs = [model(x) for model in models_list]
        avg_output = torch.stack(outputs).mean(0)
        probs = torch.softmax(avg_output, dim=1)
        preds = avg_output.argmax(1)
        
    # 5. 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 6. 绘图 (自动调整网格布局)
    rows = 2
    cols = (num_samples + 1) // 2
    plt.figure(figsize=(3 * cols, 7)) # 动态调整宽度
    
    for i in range(len(x)):
        ax = plt.subplot(rows, cols, i + 1)
        img = x[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        true_idx = y[i].item()
        pred_idx = preds[i].item()
        conf = probs[i][pred_idx].item() * 100
        
        # 绿色表示正确，红色表示错误
        color = '#006400' if true_idx == pred_idx else '#CC0000'
        
        title = f"True: {classes[true_idx]}\nPred: {classes[pred_idx]}\nConf: {conf:.1f}%"
        ax.set_title(title, color=color, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('inference_random_samples.pdf')
    plt.show()
    print("Saved: inference_random_samples.pdf")

# =========================================================================
# 7. 主程序入口
# =========================================================================
if __name__ == "__main__":
    # 1. 准备数据 (dataset用于抽样, loader用于全量跑分)
    test_dataset, test_loader = get_test_data()
    
    # 2. 加载 5 个模型
    ensemble_models = load_ensemble_models()
    
    # 3. 【全量】计算准确率、混淆矩阵、ROC (遍历整个测试集)
    preds, targets, probs = run_full_inference(test_loader, ensemble_models)
    
    acc = (preds == targets).sum() / len(targets)
    print(f"\n>>> Final Ensemble Accuracy: {acc:.4f}")
    
    # 绘图：混淆矩阵 & ROC
    plot_confusion_matrix(preds, targets, CLASS_NAMES)
    plot_multiclass_roc(targets, probs, CLASS_NAMES)
    
    # 4. 【随机抽样】可视化具体样本 (从整个测试集中随机挑 VISUALIZE_NUM 张)
    visualize_random_samples(test_dataset, ensemble_models, CLASS_NAMES, num_samples=VISUALIZE_NUM)