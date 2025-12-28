import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
from model import UNet
from utils import rgb_to_lab_tensors

# Configuration & Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
MODEL_PATH = os.path.join(root_dir, "results", "checkpoints", "best_model.pth")
TEST_REAL_DIR = os.path.join(root_dir, "data", "03_Test_Pairs", "Real")
TEST_FAKE_DIR = os.path.join(root_dir, "data", "03_Test_Pairs", "Fake")
RESULTS_DIR = os.path.join(root_dir, "results")
DETECTION_DIR = os.path.join(RESULTS_DIR, "detections")
os.makedirs(DETECTION_DIR, exist_ok=True)

# Load Model
model = UNet().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()
model.eval()

# Analysis an image and returns its average reconstruction error
def analyze_image(image_path, label, save_heatmap=False):
    img_pil = Image.open(image_path).convert('RGB').resize((256, 256))
    L, ab_orig = rgb_to_lab_tensors(img_pil)
    L = L.unsqueeze(0).to(device)
    
    with torch.no_grad():
        ab_pred = model(L).cpu().squeeze(0)
    
    error_map = torch.sqrt(torch.sum((ab_pred - ab_orig)**2, dim=0)).numpy()
    avg_error = np.mean(error_map)

    if save_heatmap:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_pil)
        axes[0].set_title(f"Suspect Image ({label})")
        axes[0].axis('off')
        
        im = axes[1].imshow(error_map, cmap='jet')
        axes[1].set_title("Chromatic Inconsistency")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        filename = f"heatmap_{label}_{os.path.basename(image_path)}"
        plt.savefig(os.path.join(DETECTION_DIR, filename))
        plt.close()
    
    return avg_error

# Evaluation Loop & ROC Calculation
if __name__ == "__main__":
    print("\n--- Phase 3: Forensic Investigation ---")
    
    scores = []
    y_true = [] # 0 for Real, 1 for Fake

    SAMPLES_TO_TEST = 50 

    # Process Real Images
    real_files = sorted([f for f in os.listdir(TEST_REAL_DIR) if f.lower().endswith(('.jpg', '.png'))])[:SAMPLES_TO_TEST]
    print(f"Analyzing {len(real_files)} Real images...")
    for i, img_name in enumerate(real_files):
        err = analyze_image(os.path.join(TEST_REAL_DIR, img_name), "Real", save_heatmap=(i < 5))
        scores.append(err)
        y_true.append(0)

    # Process Fake Images
    fake_files = sorted([f for f in os.listdir(TEST_FAKE_DIR) if f.lower().endswith(('.jpg', '.png'))])[:SAMPLES_TO_TEST]
    print(f"Analyzing {len(fake_files)} Fake images...")
    for i, img_name in enumerate(fake_files):
        err = analyze_image(os.path.join(TEST_FAKE_DIR, img_name), "Fake", save_heatmap=(i < 5))
        scores.append(err)
        y_true.append(1)

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    print(f"\nResults Analysis:")
    print(f"Detection AUC: {roc_auc:.4f}")
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Real classified as Fake)')
    plt.ylabel('True Positive Rate (Fake correctly identified)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    print(f"ROC Curve saved to: {roc_path}")
    print(f"Sample heatmaps saved in: {DETECTION_DIR}")