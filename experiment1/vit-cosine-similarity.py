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
parser = argparse.ArgumentParser(description="Generate ViT Cosine Similarity Heatmaps.")
parser.add_argument("image_path", type=str, help="Path to your input image (e.g., scan.png)")
parser.add_argument("--row", type=int, default=2, help="Row index of the target patch")
parser.add_argument("--col", type=int, default=2, help="Column index of the target patch")
args = parser.parse_args()

IMAGE_PATH = args.image_path
TARGET_PATCH_ROW = args.row
TARGET_PATCH_COL = args.col

# ==========================================
# 2. LOAD MODEL & PREPARE IMAGE
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'google/vit-base-patch32-224-in21k'

print(f"Loading image: {IMAGE_PATH}")
print(f"Targeting Patch -> Row: {TARGET_PATCH_ROW}, Col: {TARGET_PATCH_COL}")
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

if TARGET_PATCH_ROW >= num_patches_h or TARGET_PATCH_COL >= num_patches_w:
    print(f"\nERROR: Target patch out of bounds!")
    print(f"Your image generated a grid of {num_patches_h} rows and {num_patches_w} columns.")
    print(f"Maximum allowed --row is {num_patches_h - 1}")
    print(f"Maximum allowed --col is {num_patches_w - 1}")
    exit()

# ==========================================
# 3. EXTRACT EMBEDDINGS & CALCULATE SIMILARITY
# ==========================================
print("Extracting features and calculating similarity...")
with torch.no_grad():
    outputs = model(**inputs, interpolate_pos_encoding=True)
    patch_tokens = outputs.last_hidden_state[0, 1:, :] 

target_idx = (TARGET_PATCH_ROW * num_patches_w) + TARGET_PATCH_COL
target_vector = patch_tokens[target_idx] 

# Calculate Cosine Similarity
cosine_scores = F.cosine_similarity(patch_tokens, target_vector.unsqueeze(0), dim=-1)

# ==========================================
# 4. VISUALIZATION PROCESSING
# ==========================================
def process_heatmap(scores):
    score_map = scores.reshape(num_patches_h, num_patches_w).cpu().numpy()
    score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())
    
    # Changed to INTER_NEAREST to show exact, rigid patch boundaries
    resized = cv2.resize(score_map, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    
    cmap = plt.get_cmap('inferno')
    colored = (cmap(resized)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(original_img, 0.4, colored, 0.6, 0)

cosine_overlay = process_heatmap(cosine_scores)

# ==========================================
# 5. PLOT THE RESULTS
# ==========================================
print("Generating plot...")

# Calculate the bounding box coordinates for the target patch
patch_x = int((TARGET_PATCH_COL / num_patches_w) * orig_width)
patch_y = int((TARGET_PATCH_ROW / num_patches_h) * orig_height)
box_w = int(orig_width / num_patches_w)
box_h = int(orig_height / num_patches_h)

# Adjust figure size for a 2-panel layout
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
plt.gca().add_patch(plt.Rectangle((patch_x, patch_y), box_w, box_h, edgecolor='red', facecolor='none', lw=2))

# Panel 2: Cosine Similarity Overlay
ax2 = plt.subplot(1, 2, 2)
plt.title(f"Semantic Similarity Overlay\nTarget: Row {TARGET_PATCH_ROW}, Col {TARGET_PATCH_COL}")
plt.imshow(cosine_overlay)
plt.axis('off')
plt.gca().add_patch(plt.Rectangle((patch_x, patch_y), box_w, box_h, edgecolor='red', facecolor='none', lw=2))

# Attach colorbar to the heatmap panel
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Similarity Score', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()