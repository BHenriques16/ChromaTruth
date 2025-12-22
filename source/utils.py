import numpy as np
from skimage import color
import torch

# Converts RGB PIL image to normalized L and ab tensors
def rgb_to_lab_tensors(img_rgb):
    img_np = np.array(img_rgb) / 255.0
    img_lab = color.rgb2lab(img_np)
    
    # Normalize L to [-1, 1] and ab to [-1, 1]
    L = (img_lab[:, :, 0:1] / 50.0) - 1.0 
    ab = img_lab[:, :, 1:] / 128.0
    
    L = torch.from_numpy(L).permute(2, 0, 1).float()
    ab = torch.from_numpy(ab).permute(2, 0, 1).float()
    
    return L, ab