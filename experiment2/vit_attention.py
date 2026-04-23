"""
vit_attention.py
────────────────────────────────────────────────────────────────────────────
Extract raw Softmax(QK^T / √d) attention matrices from a base ViT using the
Hugging Face `transformers` library (the ONLY clean way to get them from the
C++-fused nn.MultiheadAttention backend).

Usage
─────
  # Colored region overlay — patches with similar structure share a color
  python vit_attention.py --image photo.jpg --cluster-regions
  python vit_attention.py --image photo.jpg --cluster-regions --clusters 7 --layer 11

  # CLS attention heatmap + full N×N matrix side by side
  python vit_attention.py --image photo.jpg

  # Mosaic of all 12 heads
  python vit_attention.py --image photo.jpg --all-heads --layer 11

  # Attention rollout (fuses all layers)
  python vit_attention.py --image photo.jpg --cls-rollout

Output
──────
  vit_attention_output.png  — saved in the current working directory
"""

import argparse, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

from transformers import ViTModel, ViTImageProcessor, ViTConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID   = "google/vit-base-patch16-224"
PATCH_GRID = 14          # 224 / 16
N_PATCHES  = PATCH_GRID * PATCH_GRID   # 196

def load_model():
    print(f"Loading {MODEL_ID} ...")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    # Force output_attentions on the config BEFORE loading weights.
    # Passing output_attentions=True only at forward() time is unreliable:
    # some HF versions respect the config flag and ignore the runtime kwarg,
    # leaving outputs.attentions as an empty tuple.
    config = ViTConfig.from_pretrained(MODEL_ID)
    config.output_attentions = True

    model = ViTModel.from_pretrained(MODEL_ID, config=config,
                                     ignore_mismatched_sizes=True)
    model.eval()
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Forward pass — attention weights collected via the config flag
# ─────────────────────────────────────────────────────────────────────────────
def extract_attentions(processor, model, image: Image.Image):
    """
    attentions : tuple of 12 tensors, each (1, num_heads, N+1, N+1)
                 Values = raw Softmax(QK^T / sqrt(d_k)).  N=196, +1 for [CLS].
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if not outputs.attentions:
        raise RuntimeError(
            "outputs.attentions is empty.\n"
            "Make sure you are using ViTModel (not ViTForImageClassification) "
            "and transformers >= 4.x."
        )

    return outputs.attentions, inputs["pixel_values"]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def cls_attn_map(attn_tensor, head: int) -> np.ndarray:
    """[CLS]->patch row -> (14, 14)"""
    return attn_tensor[0, head, 0, 1:].cpu().numpy().reshape(PATCH_GRID, PATCH_GRID)

def full_matrix(attn_tensor, head: int) -> np.ndarray:
    """Full (N+1)x(N+1) matrix for one head."""
    return attn_tensor[0, head].cpu().numpy()

def patch_similarity_features(attn_tensor) -> np.ndarray:
    """
    Build a feature vector for each of the 196 patch tokens using their
    OUTGOING attention rows from the patch-to-patch submatrix
    (rows 1:, cols 1: — excluding [CLS]).

    Each patch i attends to all other patches j according to Softmax(QK^T).
    That 196-dim row is the patch's structural fingerprint: patches with
    similar texture/structure attend similarly to their context.

    We average across all heads so clustering uses the consensus signal
    rather than one head's idiosyncratic view.

    Returns  features : (196, 196) float32
    """
    avg = attn_tensor[0].mean(dim=0).cpu().numpy()   # (197, 197)
    return avg[1:, 1:].astype(np.float32)             # (196, 196)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Attention rollout
# ─────────────────────────────────────────────────────────────────────────────
def attention_rollout(attentions) -> np.ndarray:
    rollout = torch.eye(N_PATCHES + 1)
    for attn in attentions:
        avg = attn[0].mean(dim=0).cpu()
        avg = avg + torch.eye(avg.shape[0])
        avg = avg / avg.sum(dim=-1, keepdim=True)
        rollout = avg @ rollout
    return rollout[0, 1:].numpy().reshape(PATCH_GRID, PATCH_GRID)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  k-means clustering (pure numpy — no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_numpy(X: np.ndarray, k: int, n_iter: int = 150, n_init: int = 8,
                 seed: int = 0) -> np.ndarray:
    """
    k-means++ initialisation + Lloyd iterations.
    Runs n_init restarts, returns labels for the run with lowest inertia.
    """
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf

    for _ in range(n_init):
        # k-means++ init
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            dists = np.min(
                np.stack([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0
            )
            probs = dists / dists.sum()
            centers.append(X[rng.choice(len(X), p=probs)])
        centers = np.stack(centers)   # (k, D)

        labels = np.zeros(len(X), dtype=int)
        for _ in range(n_iter):
            dists      = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)

        inertia = sum(
            np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()

    return best_labels   # (196,)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
# Perceptually distinct palette (12 colours)
REGION_COLORS = [
    (0.25, 0.60, 0.95),   # blue
    (0.95, 0.40, 0.25),   # coral
    (0.30, 0.85, 0.55),   # green
    (0.95, 0.75, 0.15),   # amber
    (0.75, 0.35, 0.90),   # violet
    (0.95, 0.35, 0.65),   # pink
    (0.20, 0.85, 0.90),   # cyan
    (0.90, 0.55, 0.20),   # orange
    (0.50, 0.90, 0.30),   # lime
    (0.40, 0.30, 0.85),   # indigo
    (0.85, 0.20, 0.30),   # red
    (0.20, 0.65, 0.55),   # teal
]

def normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)

def overlay_heatmap(ax, base_img: np.ndarray, heatmap: np.ndarray,
                    alpha=0.55, cmap="inferno", title=""):
    h, w = base_img.shape[:2]
    hm_up = np.array(
        Image.fromarray((normalise(heatmap) * 255).astype(np.uint8))
        .resize((w, h), Image.BILINEAR)
    ) / 255.0
    ax.imshow(base_img)
    ax.imshow(hm_up, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")


def build_region_overlay(labels: np.ndarray, img_h: int, img_w: int,
                          alpha: float = 0.55) -> np.ndarray:
    """
    labels : (196,) int cluster IDs for the 14x14 patch tokens.
    Returns RGBA overlay at (img_h, img_w, 4), upscaled with NEAREST
    so patch boundaries stay crisp rather than bleeding into each other.
    """
    label_grid = labels.reshape(PATCH_GRID, PATCH_GRID)
    k = labels.max() + 1
    rgba_small = np.zeros((PATCH_GRID, PATCH_GRID, 4), dtype=np.float32)
    for cid in range(k):
        r, g, b = REGION_COLORS[cid % len(REGION_COLORS)]
        mask = label_grid == cid
        rgba_small[mask, 0] = r
        rgba_small[mask, 1] = g
        rgba_small[mask, 2] = b
        rgba_small[mask, 3] = alpha
    rgba_pil = Image.fromarray((rgba_small * 255).astype(np.uint8), mode="RGBA")
    return np.array(rgba_pil.resize((img_w, img_h), Image.NEAREST)) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Figure: clustered region overlay
# ─────────────────────────────────────────────────────────────────────────────
def fig_cluster_regions(img_np, attentions, layer, k, alpha, out_path):
    """
    For each of the 196 patch tokens, use its row from Softmax(QK^T)
    (averaged across heads) as a structural fingerprint, then k-means
    cluster those 196-dim vectors.  Same color = structurally similar context.
    """
    print(f"  Extracting patch features from layer {layer} ...")
    features = patch_similarity_features(attentions[layer])   # (196, 196)

    print(f"  Running k-means  k={k} ...")
    labels = kmeans_numpy(features, k=k)

    img_h, img_w = img_np.shape[:2]
    ar = img_h / img_w
    panel_w = 5.5
    panel_h = max(3.5, panel_w * ar)

    fig, axes = plt.subplots(1, 2,
                             figsize=(panel_w * 2 + 0.3, panel_h + 0.8),
                             facecolor="#0e0e0e")

    axes[0].imshow(img_np)
    axes[0].set_title("Input image", color="white", fontsize=10)
    axes[0].axis("off")

    overlay = build_region_overlay(labels, img_h, img_w, alpha=alpha)
    axes[1].imshow(img_np)
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Structural regions  k={k}  layer {layer}  Softmax(QKt) k-means",
        color="white", fontsize=9
    )
    axes[1].axis("off")

    patches = [
        mpatches.Patch(color=REGION_COLORS[i % len(REGION_COLORS)],
                       label=f"region {i}")
        for i in range(k)
    ]
    axes[1].legend(
        handles=patches, loc="lower right", fontsize=7,
        framealpha=0.55, facecolor="#111", labelcolor="white",
        edgecolor="#444", ncol=min(k, 4),
    )

    fig.suptitle(
        "ViT-Base/16  patch structural similarity  k-means on Softmax(QKt) rows",
        color="white", fontsize=11, y=1.01
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Existing figure modes
# ─────────────────────────────────────────────────────────────────────────────
CMAP_HEADS = [
    "inferno", "viridis", "magma", "plasma",
    "cividis", "hot", "YlOrRd", "Blues",
    "Greens", "PuRd", "BuPu", "copper",
]

def fig_single(img_np, attentions, layer, head, out_path):
    attn = attentions[layer]
    hm   = cls_attn_map(attn, head)
    full = full_matrix(attn, head)

    img_h, img_w = img_np.shape[:2]
    ar      = img_h / img_w
    panel_w = 4.5
    panel_h = max(3.0, panel_w * ar)
    total_w = panel_w * 2 + panel_h + 0.8
    fig = plt.figure(figsize=(total_w, panel_h + 0.7), facecolor="#0e0e0e")
    gs  = GridSpec(1, 3, figure=fig, wspace=0.08,
                   left=0.03, right=0.97, top=0.88, bottom=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_np); ax0.axis("off")
    ax0.set_title("Input image", color="white", fontsize=10)

    ax1 = fig.add_subplot(gs[1])
    overlay_heatmap(ax1, img_np, hm, cmap="inferno",
                    title=f"[CLS] attention  layer {layer}  head {head}")
    ax1.title.set_color("white")

    ax2 = fig.add_subplot(gs[2])
    im = ax2.imshow(full, cmap="magma", aspect="auto")
    ax2.set_title(f"Softmax(QKt)  {full.shape[0]}x{full.shape[0]}",
                  color="white", fontsize=10)
    ax2.set_xlabel("Key token idx", color="#aaa", fontsize=8)
    ax2.set_ylabel("Query token idx", color="#aaa", fontsize=8)
    ax2.tick_params(colors="#666", labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#888", labelsize=7)
    cb.outline.set_edgecolor("#333")

    fig.suptitle(f"ViT-Base/16  Softmax(QKt)  Layer {layer}  Head {head}",
                 color="white", fontsize=12, y=0.97)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path}")


def fig_all_heads(img_np, attentions, layer, out_path):
    attn  = attentions[layer]
    nhead = attn.shape[1]
    cols  = 4
    rows  = (nhead + cols - 1) // cols
    img_h, img_w = img_np.shape[:2]
    ar = img_h / img_w
    panel_w, panel_h = 3.5, max(2.5, 3.5 * ar)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * panel_w, rows * panel_h + 0.6),
                             facecolor="#0e0e0e")
    axes = axes.flatten()
    for h in range(nhead):
        overlay_heatmap(axes[h], img_np, cls_attn_map(attn, h),
                        cmap=CMAP_HEADS[h % len(CMAP_HEADS)], title=f"head {h}")
        axes[h].title.set_color("white")
    for ax in axes[nhead:]:
        ax.axis("off")
    fig.suptitle(
        f"ViT-Base/16  All {nhead} heads  Layer {layer}  [CLS]->patch attention",
        color="white", fontsize=11, y=1.01
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path}")


def fig_rollout(img_np, attentions, out_path):
    hm = attention_rollout(attentions)
    img_h, img_w = img_np.shape[:2]
    ar = img_h / img_w
    panel_w, panel_h = 5.0, max(3.0, 5.0 * ar)
    fig, axes = plt.subplots(1, 2,
                             figsize=(panel_w * 2 + 0.4, panel_h + 0.5),
                             facecolor="#0e0e0e")
    axes[0].imshow(img_np); axes[0].axis("off")
    axes[0].set_title("Input image", color="white", fontsize=10)
    overlay_heatmap(axes[1], img_np, hm, cmap="turbo",
                    title="Attention rollout  (all layers, avg heads)")
    axes[1].title.set_color("white")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Softmax(QK^T) from ViT-Base and visualise attention maps."
    )
    p.add_argument("--image",    required=True, help="Path to input image")
    p.add_argument("--layer",    type=int, default=11,
                   help="Transformer layer index 0-11  (default: 11)")
    p.add_argument("--head",     type=int, default=0,
                   help="Attention head index 0-11  (default: 0)")
    p.add_argument("--clusters", type=int, default=5,
                   help="Number of region clusters for --cluster-regions  (default: 5)")
    p.add_argument("--alpha",    type=float, default=0.55,
                   help="Overlay transparency 0-1  (default: 0.55)")
    p.add_argument("--cluster-regions", action="store_true",
                   help="Cluster patches by structural similarity -> colored region overlay")
    p.add_argument("--all-heads", action="store_true",
                   help="Mosaic of all 12 heads for --layer")
    p.add_argument("--cls-rollout", action="store_true",
                   help="Attention rollout  (Abnar & Zuidema 2020)")
    p.add_argument("--output",   default="vit_attention_output.png",
                   help="Output PNG path  (default: vit_attention_output.png)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        pil_img = Image.open(args.image).convert("RGB")
    except Exception as e:
        sys.exit(f"Cannot open image: {e}")

    img_np = np.array(pil_img)   # original resolution and aspect ratio

    processor, model = load_model()
    attentions, _    = extract_attentions(processor, model, pil_img)

    num_layers = len(attentions)
    num_heads  = attentions[0].shape[1]
    seq_len    = attentions[0].shape[2]
    orig_h, orig_w = img_np.shape[:2]

    print(f"\nImage       : {orig_w}x{orig_h} px  (original, preserved for display)")
    print(f"Model input : 224x224 px  (processor resizes internally)")
    print(f"Layers      : {num_layers}")
    print(f"Heads       : {num_heads}")
    print(f"Seq len     : {seq_len}  ({seq_len - 1} patch tokens + 1 [CLS])")
    print(f"Matrix      : {seq_len}x{seq_len} per head per layer\n")

    layer = min(args.layer,    num_layers - 1)
    head  = min(args.head,     num_heads  - 1)
    k     = max(2, min(args.clusters, len(REGION_COLORS)))

    if args.cluster_regions:
        fig_cluster_regions(img_np, attentions, layer, k, args.alpha, args.output)
    elif args.cls_rollout:
        fig_rollout(img_np, attentions, args.output)
    elif args.all_heads:
        fig_all_heads(img_np, attentions, layer, args.output)
    else:
        fig_single(img_np, attentions, layer, head, args.output)


if __name__ == "__main__":
    main()
