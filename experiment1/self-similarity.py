import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

# ==========================================
# 1. COMMAND LINE ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Generate ViT Global Self-Similarity Heatmap.")
parser.add_argument("image_path", type=str, help="Path to your input image (e.g., scan.png)")
args = parser.parse_args()

IMAGE_PATH = args.image_path

# ==========================================
# 2. LOAD MODEL & PREPARE IMAGE
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'google/vit-base-patch32-224-in21k'

print(f"Loading image: {IMAGE_PATH}")
print("Loading ViT-Base (patch32)...")

model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
PATCH_SIZE = model.config.patch_size

try:
    image_pil = Image.open(IMAGE_PATH).convert('RGB')
except FileNotFoundError:
    print(f"\nERROR: Could not find the image file at '{IMAGE_PATH}'. Please check the path and try again.")
    exit()

original_img = np.array(image_pil)
orig_width, orig_height = image_pil.size

# Snap dimensions to multiples of ViT patch size (e.g. 32 for patch32)
new_width = max(PATCH_SIZE, orig_width - (orig_width % PATCH_SIZE))
new_height = max(PATCH_SIZE, orig_height - (orig_height % PATCH_SIZE))

processor = ViTImageProcessor.from_pretrained(MODEL_NAME, size={"height": new_height, "width": new_width})
inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)

num_patches_w = new_width // PATCH_SIZE
num_patches_h = new_height // PATCH_SIZE
print(f"Grid Size: {num_patches_w} columns by {num_patches_h} rows (patch size {PATCH_SIZE}).")

# ==========================================
# 3. EXTRACT EMBEDDINGS & CALCULATE SELF-SIMILARITY
# ==========================================
print("Extracting features and computing N x N similarity matrix...")
with torch.no_grad():
    outputs = model(**inputs, interpolate_pos_encoding=True)
    # Shape: (Total Patches, 768)
    patch_tokens = outputs.last_hidden_state[0, 1:, :] 

# Step A: L2 Normalize all tokens
normalized_tokens = F.normalize(patch_tokens, p=2, dim=-1)

# Step B: Matrix Multiplication (Every patch compared to every other patch)
# Shape: (Total Patches, Total Patches)
all_vs_all_matrix = torch.matmul(normalized_tokens, normalized_tokens.T)

# Step C: Average the scores across all patches to find Global Typicality
# Shape: (Total Patches,)
average_similarity_scores = all_vs_all_matrix.mean(dim=0)

# ==========================================
# 4. VISUALIZATION PROCESSING
# ==========================================
def process_heatmap(scores):
    score_map = scores.reshape(num_patches_h, num_patches_w).cpu().numpy()
    score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())
    
    # Using Nearest Neighbor for rigid square blocks
    resized = cv2.resize(score_map, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    
    cmap = plt.get_cmap('inferno')
    colored = (cmap(resized)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(original_img, 0.4, colored, 0.6, 0)

anomaly_overlay = process_heatmap(average_similarity_scores)

# ==========================================
# 5. PLOT THE RESULTS
# ==========================================
print("Generating plot...")

fig = plt.figure(figsize=(10, 14)) 

# Setup the colorbar reference scale (0.0 to 1.0 using Inferno)
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
sm = cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])

# Panel 1: Original Scan
ax1 = plt.subplot(1, 2, 1)
plt.title("Original Scan")
plt.imshow(original_img)
plt.axis('off')

# Panel 2: Self-Similarity (Anomaly) Overlay
ax2 = plt.subplot(1, 2, 2)
plt.title("Global Self-Similarity Map\n(Dark = Anomaly, Bright = Typical)")
plt.imshow(anomaly_overlay)
plt.axis('off')

# Attach colorbar
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Average Self-Similarity (Normalized)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()