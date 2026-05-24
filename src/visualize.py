"""
Görselleştirme Modülü
Üretilen görseller: outputs/figures/ klasörüne kaydedilir.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures"
)

def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def _save(fig, filename):
    _ensure_dir()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Kaydedildi] {path}")
    return path


# ─────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, dataset_name, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomali"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {dataset_name} / {model_name}", fontsize=12)
    return _save(fig, f"confusion_matrix_{dataset_name}_{model_name}.png")


# ─────────────────────────────────────────────
# 2. Transition Probability Heatmap
# ─────────────────────────────────────────────
def plot_transition_heatmap(transition_matrix, alphabet_size, dataset_name):
    labels = [chr(97 + i) for i in range(alphabet_size)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(transition_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Geçiş Olasılığı")
    ax.set_xticks(range(alphabet_size))
    ax.set_yticks(range(alphabet_size))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hedef Durum")
    ax.set_ylabel("Kaynak Durum")
    ax.set_title(f"Transition Probability Heatmap — {dataset_name}", fontsize=12)
    for i in range(alphabet_size):
        for j in range(alphabet_size):
            ax.text(j, i, f"{transition_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if transition_matrix[i, j] < 0.6 else "white")
    return _save(fig, f"transition_heatmap_{dataset_name}.png")


# ─────────────────────────────────────────────
# 3. Automata State Diagram
# ─────────────────────────────────────────────
def plot_state_diagram(transition_matrix, alphabet_size, dataset_name, top_n=12):
    """
    En yüksek olasılıklı top_n geçişi ok olarak çizer.
    Durumlar daire üzerinde eşit açılarla yerleştirilir.
    """
    labels = [chr(97 + i) for i in range(alphabet_size)]
    n = alphabet_size

    # Düğüm konumları — daire üzerinde
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {labels[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

    # En yüksek top_n geçişi seç (öz-döngüler hariç)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((labels[i], labels[j], transition_matrix[i, j]))
    edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges[:top_n]

    max_prob = max(e[2] for e in top_edges) if top_edges else 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Automata State Diagram — {dataset_name}\n(Top {top_n} geçiş)", fontsize=12)

    # Oklar
    for src, dst, prob in top_edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        alpha = 0.3 + 0.7 * (prob / max_prob)
        lw = 0.5 + 3.0 * (prob / max_prob)
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=plt.cm.YlOrRd(prob / max_prob),
                lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.15"
            )
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"{prob:.2f}", fontsize=6, ha="center", va="center", alpha=0.7)

    # Düğümler
    node_r = 0.13
    for label, (x, y) in pos.items():
        circle = plt.Circle((x, y), node_r, color="#4A90D9", zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    return _save(fig, f"state_diagram_{dataset_name}.png")


# ─────────────────────────────────────────────
# 4. Parametre Duyarlılık Grafikleri
# ─────────────────────────────────────────────
def plot_parameter_sensitivity(grid_results, dataset_name):
    """
    grid_results: [{"w_size": int, "a_size": int, "F1": float}, ...]
    """
    w_sizes = sorted(set(r["w_size"] for r in grid_results))
    a_sizes = sorted(set(r["a_size"] for r in grid_results))

    # Matris oluştur
    matrix = np.zeros((len(w_sizes), len(a_sizes)))
    for r in grid_results:
        wi = w_sizes.index(r["w_size"])
        ai = a_sizes.index(r["a_size"])
        matrix[wi, ai] = r["F1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Heatmap
    im = axes[0].imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=axes[0], label="F1-Score")
    axes[0].set_xticks(range(len(a_sizes)))
    axes[0].set_yticks(range(len(w_sizes)))
    axes[0].set_xticklabels(a_sizes)
    axes[0].set_yticklabels(w_sizes)
    axes[0].set_xlabel("Alphabet Size")
    axes[0].set_ylabel("Window Size")
    axes[0].set_title(f"Parametre Duyarlılık Heatmap — {dataset_name}")
    for wi in range(len(w_sizes)):
        for ai in range(len(a_sizes)):
            axes[0].text(ai, wi, f"{matrix[wi, ai]:.3f}",
                         ha="center", va="center", fontsize=8, color="white")

    # Çizgi grafik — window size etkisi
    for ai, a in enumerate(a_sizes):
        f1_vals = [matrix[wi, ai] for wi in range(len(w_sizes))]
        axes[1].plot(w_sizes, f1_vals, marker="o", label=f"alphabet={a}")
    axes[1].set_xlabel("Window Size")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title(f"Window Size Etkisi — {dataset_name}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, f"parameter_sensitivity_{dataset_name}.png")


# ─────────────────────────────────────────────
# 5. ROC ve Precision-Recall Eğrileri
# ─────────────────────────────────────────────
def plot_roc_pr(y_true, y_scores, dataset_name, model_name):
    """
    y_scores: anomali olasılıkları (yüksek = anomali)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#4A90D9", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Eğrisi — {dataset_name} / {model_name}")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    axes[1].plot(recall, precision, color="#E07B54", lw=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Eğrisi — {dataset_name} / {model_name}")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, f"roc_pr_{dataset_name}_{model_name}.png")


# ─────────────────────────────────────────────
# 6. Model Karşılaştırma Bar Chart
# ─────────────────────────────────────────────
def plot_model_comparison(results_dict, dataset_name):
    """
    results_dict: {"ModelAdı": {"F1": float, "Std": float}, ...}
    """
    models = list(results_dict.keys())
    f1_vals = [results_dict[m]["F1"] for m in models]
    std_vals = [results_dict[m].get("Std", 0) for m in models]

    colors = ["#4A90D9", "#E07B54", "#5BAD6F", "#9B59B6"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, f1_vals, yerr=std_vals, capsize=5,
                  color=colors[:len(models)], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, min(1.0, max(f1_vals) * 1.3 + 0.05))
    ax.set_title(f"Model Performans Karşılaştırması — {dataset_name}")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val, std in zip(bars, f1_vals, std_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, f"model_comparison_{dataset_name}.png")