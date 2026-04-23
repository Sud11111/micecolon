# ViT Attention Extractor

Extract raw **Softmax(QKᵀ / √d_k)** attention matrices from `google/vit-base-patch16-224`
and visualise them as spatial overlays on your image.

---

## The PyTorch Engineering Quirk

The standard `torchvision` ViT uses PyTorch's C++ fused `nn.MultiheadAttention` kernel.
For memory efficiency, this kernel **discards attention weights** after the forward pass,
making forward hooks useless:

```python
# ❌ This will NOT work — hook fires but attn_weights is None
def hook(module, input, output):
    print(output[1])   # always None for fused kernel

model.encoder.layers[11].self_attn.register_forward_hook(hook)
```

The fix: use HuggingFace `transformers`, which implements ViT in pure Python/PyTorch
and exposes `output_attentions=True`:

```python
# ✅ This works — clean, no hacks required
from transformers import ViTModel, ViTImageProcessor

model = ViTModel.from_pretrained("google/vit-base-patch16-224")
outputs = model(**inputs, output_attentions=True)

# outputs.attentions: tuple of 12 tensors
# each shape: (batch, num_heads, N+1, N+1)
# N = 196  (14×14 patches),  +1 for [CLS] token
```

---

## Installation

```bash
pip install transformers torch torchvision pillow matplotlib numpy
```

---

## Usage

### Single head — [CLS] overlay + full N×N matrix heatmap
```bash
python vit_attention.py --image photo.jpg
```

### Choose a specific layer and head
```bash
python vit_attention.py --image photo.jpg --layer 11 --head 5
```

### Mosaic of all 12 heads for a given layer
```bash
python vit_attention.py --image photo.jpg --all-heads --layer 11
```

### Attention rollout (fuses all layers, all heads)
```bash
python vit_attention.py --image photo.jpg --cls-rollout
```

### Custom output path
```bash
python vit_attention.py --image photo.jpg --output my_result.png
```

---

## What each output shows

| Mode | What you see |
|---|---|
| Default (single head) | Left: input image · Centre: [CLS]→patch attention overlay · Right: full (N+1)×(N+1) Softmax(QKᵀ) heatmap |
| `--all-heads` | 12-panel mosaic, each head with its own colormap — reveals head specialisation |
| `--cls-rollout` | Rolled-up signal tracing how information flows from patches to [CLS] across all layers |

---

## Anatomy of the attention tensor

```
outputs.attentions[layer]   shape: (1, 12, 197, 197)
                                        │    │    │
                                   heads │    │    └── key tokens  (196 patches + [CLS])
                                         └────┘─── query tokens
```

Row 0 = `[CLS]` token. Rows 1–196 = the 14×14 = 196 patch tokens (left-to-right, top-to-bottom).

The matrix value at `[q, k]` is the fraction of attention query `q` pays to key `k`
after softmax — a proper probability distribution summing to 1 along the key axis.

---

## Layers & head behaviour (ViT-Base)

| Layer range | Typical behaviour |
|---|---|
| 0–3 | Low-level texture, edges, local structure |
| 4–7 | Mid-level parts, object boundaries |
| 8–11 | Semantic regions, global context; layer 11 head 0 often shows sharp [CLS] focus |

Heads specialise — some attend to global structure, others to fine local patches.
`--all-heads` on layer 11 is the most revealing visualisation.
