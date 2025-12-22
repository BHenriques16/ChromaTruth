import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNet
from utils import rgb_to_lab_tensors

# Dataset Class with Subsampling Support
# Loads real face images and prepares L (Input) and ab (Target) channels.
class ChromaDataset(Dataset):
    def __init__(self, folder_path, max_samples=None):
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        # List and sort files for consistency
        all_files = sorted([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Subsampling logic to speed up training
        if max_samples is not None:
            self.image_files = all_files[:max_samples]
        else:
            self.image_files = all_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img_rgb = Image.open(img_path).convert('RGB').resize((256, 256))
        
        # Convert to Lab tensors using utils.py
        L, ab = rgb_to_lab_tensors(img_rgb)
        return L, ab

# Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for L, ab_real in dataloader:
            L, ab_real = L.to(device), ab_real.to(device)
            ab_pred = model(L)
            loss = criterion(ab_pred, ab_real)
            val_loss += loss.item()
    return val_loss / len(dataloader)

#   MAIN TRAINING LOOP
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    TRAIN_DIR = os.path.join(root_dir, "data", "01_Real", "train")
    VAL_DIR = os.path.join(root_dir, "data", "01_Real", "val")
    RESULTS_DIR = os.path.join(root_dir, "results")
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Cponfigurations
    MAX_SAMPLES = 3000 
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 20
    PATIENCE = 5  # Early stopping limit

    # Initialize Model, Loss, and Optimizer
    model = UNet().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load Data
    train_set = ChromaDataset(TRAIN_DIR, max_samples=MAX_SAMPLES)
    val_set = ChromaDataset(VAL_DIR, max_samples=800)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset Size -> Train: {len(train_set)} | Val: {len(val_set)}")

    # Tracking metrics
    history_train = []
    history_val = []
    best_loss = float('inf')
    no_improve_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        # Training progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        for L, ab_real in pbar:
            L, ab_real = L.to(device), ab_real.to(device)
            
            optimizer.zero_grad()
            ab_pred = model(L)
            loss = criterion(ab_pred, ab_real)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Evaluate and log
        avg_train = running_train_loss / len(train_loader)
        avg_val = validate(model, val_loader, criterion, device)
        
        history_train.append(avg_train)
        history_val.append(avg_val)

        print(f"Summary -> Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # Update Plot
        plt.figure(figsize=(10, 5))
        plt.plot(history_train, label='Train MSE')
        plt.plot(history_val, label='Val MSE')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Learning Curve (Phase 2)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"))
        plt.close()

        # Early Stopping & Checkpointing
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            no_improve_counter = 0
            print("Best model saved!")
        else:
            no_improve_counter += 1
            print(f"No improvement for {no_improve_counter} epochs.")

        if no_improve_counter >= PATIENCE:
            print(f"\nEARLY STOPPING triggered at epoch {epoch+1}")
            break

    print(f"\nProcess Complete. Best weights saved in: {CHECKPOINT_DIR}/best_model.pth")