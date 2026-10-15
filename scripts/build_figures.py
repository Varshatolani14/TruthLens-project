"""Create simple architecture/workflow figures used by the report and PPT."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fill):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.02", linewidth=1.3, edgecolor="#243447", facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="#16212b", wrap=True)


def architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    box(ax, 0.03, 0.38, 0.16, 0.24, "News article\ntext", "#DCEEFF")
    box(ax, 0.25, 0.38, 0.16, 0.24, "Cleaning +\nvalidation", "#E8F3E8")
    box(ax, 0.47, 0.57, 0.17, 0.24, "TF-IDF\nvectorizer", "#FFF0D9")
    box(ax, 0.70, 0.57, 0.19, 0.24, "Logistic\nRegression", "#F5E4F7")
    box(ax, 0.47, 0.17, 0.17, 0.24, "DistilBERT\ntokenizer", "#FFF0D9")
    box(ax, 0.70, 0.17, 0.19, 0.24, "Transformer\nclassifier", "#F5E4F7")
    box(ax, 0.92, 0.38, 0.06, 0.24, "Result", "#E6E6E6")
    arrows = [
        ((0.19, 0.50), (0.25, 0.50)),
        ((0.41, 0.50), (0.47, 0.69)),
        ((0.64, 0.69), (0.70, 0.69)),
        ((0.89, 0.69), (0.92, 0.50)),
        ((0.41, 0.50), (0.47, 0.29)),
        ((0.64, 0.29), (0.70, 0.29)),
        ((0.89, 0.29), (0.92, 0.50)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#243447"))
    ax.text(0.58, 0.90, "Traditional baseline", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.58, 0.07, "Transformer approach", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    labels = ["Dataset", "Pre-processing", "Split", "Baseline", "DistilBERT", "Evaluation", "LIME + App"]
    fills = ["#DCEEFF", "#E8F3E8", "#FFF0D9", "#F5E4F7", "#F5E4F7", "#E6E6E6", "#DCEEFF"]
    xs = [0.02, 0.16, 0.30, 0.44, 0.58, 0.72, 0.86]
    for i, (x, label, fill) in enumerate(zip(xs, labels, fills)):
        box(ax, x, 0.36, 0.10, 0.25, label, fill)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1], 0.485), xytext=(x + 0.10, 0.485), arrowprops=dict(arrowstyle="->", lw=1.4, color="#243447"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "workflow.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    workflow()
