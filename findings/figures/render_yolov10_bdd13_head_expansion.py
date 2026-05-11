#!/usr/bin/env python3
"""Render the YOLOv10s 10-class -> 13-class head expansion figure for Dr. Moradi.

Produces:
    yolov10-bdd13-head-expansion.png   (side-by-side before/after pipeline)
    yolov10-bdd13-head-detail.png      (zoom on the v10Detect head only)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent

# Color palette
C_PRESERVED = "#A8D5BA"   # green — preserved from BDD-finetune.pt
C_REINIT    = "#F4B084"   # orange — reinitialized
C_NEW       = "#E48F8F"   # red — new classes
C_FROZEN    = "#BBBBBB"
C_TEXT      = "#222222"


def block(ax, x, y, w, h, label, color=C_PRESERVED, fontsize=9, edge=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=edge if edge else "#333",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, color=C_TEXT)


def arrow(ax, x1, y1, x2, y2, label=None, fontsize=8):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color="#444", linewidth=1.2,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label,
                ha="center", va="bottom", fontsize=fontsize, color="#444")


# =====================================================================
# Figure 1: pipeline before/after, side-by-side
# =====================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 7))

for ax, title, n_classes, color_head, head_label, n_transferred in [
    (axL, "Before — yolov10-bdd-finetune.pt (10 classes)",
     10, C_PRESERVED, "v10Detect\nhead\n10 cls × 3 scales\n(607 / 619 items)", None),
    (axR, "After — yolov10s-bdd13.pt (13 classes)",
     13, C_REINIT, "v10Detect\nhead\n13 cls × 3 scales\n(12 items reinitialized)", "Transferred 607/619"),
]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    # Vertical pipeline
    block(ax, 3.5, 8.6, 3.0, 0.7, "Input image  3×640×640", color="#E0E0E0")
    arrow(ax, 5.0, 8.5, 5.0, 8.0)

    block(ax, 2.0, 6.5, 6.0, 1.4,
          "CSPDarknet backbone\n(C2f / SCDown blocks)\n~5.7 M params",
          color=C_PRESERVED)
    arrow(ax, 5.0, 6.4, 5.0, 5.9)

    block(ax, 2.5, 4.5, 5.0, 1.2,
          "PAN-FPN neck\n(P3, P4, P5  →  128, 256, 512 ch)\n~1.7 M params",
          color=C_PRESERVED)
    arrow(ax, 5.0, 4.4, 5.0, 3.9)

    block(ax, 2.0, 1.8, 6.0, 2.0, head_label, color=color_head)

    if n_transferred:
        ax.annotate(
            n_transferred,
            xy=(8.0, 7.2),
            xytext=(9.4, 7.2),
            ha="right", va="center",
            fontsize=10, color="#1a6b3a",
            fontweight="bold",
        )
        ax.annotate(
            "items preserved",
            xy=(8.0, 6.85),
            xytext=(9.4, 6.85),
            ha="right", va="center",
            fontsize=9, color="#1a6b3a",
        )
        ax.annotate(
            "12 items re-initialized",
            xy=(8.0, 2.6),
            xytext=(9.4, 2.6),
            ha="right", va="center",
            fontsize=9, color="#a85e2c",
        )

    # Class slot strip
    n_total_visible = 13
    cell_w = 5.0 / n_total_visible
    y = 0.7
    for i in range(n_total_visible):
        if i < n_classes:
            face = C_PRESERVED if ax is axL else (C_REINIT if i < 10 else C_NEW)
        else:
            face = "#FFFFFF"
        ax.add_patch(Rectangle(
            (2.5 + i * cell_w, y), cell_w, 0.5,
            facecolor=face, edgecolor="#555",
            linewidth=0.6,
        ))
    ax.text(5.0, y - 0.3, f"{n_classes} class output channels",
            ha="center", va="top", fontsize=9, fontweight="bold")

# Single legend below
fig.text(0.5, 0.04,
         "■ Preserved from BDD checkpoint     "
         "■ Reinitialized (head reset for new nc)     "
         "■ New classes (deer, cone, barrier)",
         ha="center", va="bottom", fontsize=10)
fig.text(0.5, 0.005,
         "via Ultralytics:  YOLO('yolov10-bdd-finetune.pt').train(data='bdd13.yaml', ...)  →  Overriding model.yaml nc=10 with nc=13",
         ha="center", va="bottom", fontsize=9, color="#444",
         family="monospace")

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = OUT / "yolov10-bdd13-head-expansion.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"[saved] {out_path}")
plt.close(fig)


# =====================================================================
# Figure 2: head detail — class output channel expansion
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("v10Detect head: class output expansion (10 → 13)",
             fontsize=13, fontweight="bold", pad=10)

# Three FPN scales as columns
scales = [("P3 / 80×80", 128, 4.5), ("P4 / 40×40", 256, 2.5), ("P5 / 20×20", 512, 0.5)]
for i, (scale_label, in_ch, y_off) in enumerate(scales):
    x_col = 0.5
    block(ax, x_col, y_off, 1.6, 1.2,
          f"{scale_label}\n{in_ch} ch", color="#D7E5F1")
    arrow(ax, 2.1, y_off + 0.6, 2.7, y_off + 0.6)

    # reg head (preserved)
    block(ax, 2.7, y_off + 0.7, 2.4, 0.55,
          "reg branch — 4 × reg_max ch (preserved)",
          color=C_PRESERVED, fontsize=8)
    # cls head (reinitialized)
    block(ax, 2.7, y_off + 0.05, 2.4, 0.55,
          "cls branch — Conv2d(_, NC) — REINITIALIZED",
          color=C_REINIT, fontsize=8)

    arrow(ax, 5.1, y_off + 0.6, 5.7, y_off + 0.6)

    # output channels
    block(ax, 5.7, y_off, 1.6, 1.2,
          f"reg out: 64 ch\ncls out:  10 → 13 ch", color="#FFFFFF", fontsize=9,
          edge="#888")

    arrow(ax, 7.3, y_off + 0.6, 7.8, y_off + 0.6)

    # NMS-free decoded predictions
    block(ax, 7.8, y_off, 3.8, 1.2,
          "decoded boxes  (NMS-free, end-to-end)\nclass scores: 13-vector per anchor",
          color="#F2EAD3", fontsize=9)

# Bottom legend
fig.text(0.5, 0.02,
         "Only the cls output Conv2d (3 layers, ~12 weight tensors total) is reinitialized. "
         "Reg branch + DFL + backbone + neck preserved.",
         ha="center", va="bottom", fontsize=10, color="#222")

plt.tight_layout(rect=[0, 0.04, 1, 1])
out_path2 = OUT / "yolov10-bdd13-head-detail.png"
plt.savefig(out_path2, dpi=180, bbox_inches="tight", facecolor="white")
print(f"[saved] {out_path2}")
plt.close(fig)
