"""
visualize.py
------------
Génère les 3 graphiques de l'étude empirique à partir de results.csv
et metrics_summary.json.

Graphiques produits :
    1. barplot_scores.png    — scores par langue et par domaine
    2. heatmap_robustness.png — heatmap des réponses correctes par item et langue
    3. histogram_R.png       — distribution de l'indice de robustesse R

Prérequis :
    pip install matplotlib pandas seaborn

Usage :
    python visualize.py
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
    import seaborn as sns
    import numpy as np
except ImportError:
    print("[ERREUR] Librairies manquantes.")
    print("Installe-les avec : pip install matplotlib pandas seaborn numpy")
    exit(1)

RESULTS_FILE  = "results.csv"
SUMMARY_FILE  = "metrics_summary.json"
LANGUAGES     = ["en", "fr", "ar"]
LANG_LABELS   = {"en": "Anglais (EN)", "fr": "Français (FR)", "ar": "Arabe (AR)"}
DOMAIN_LABELS = {
    "us_history":      "Histoire américaine",
    "moral_philosophy": "Philosophie morale",
    "mathematics":     "Mathématiques"
}
COLORS = {
    "en": "#2E86AB",
    "fr": "#A23B72",
    "ar": "#F18F01"
}

# Style global
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    150
})


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_results() -> list[dict]:
    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for lang in LANGUAGES:
                row[f"correct_{lang}"] = int(row[f"correct_{lang}"])
            row["robustness_R"] = float(row["robustness_R"])
            rows.append(row)
    return rows


def load_summary() -> dict:
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graphique 1 — Scores par langue et par domaine
# ---------------------------------------------------------------------------

def plot_scores_by_domain(results: list[dict], summary: dict):
    domains = list(DOMAIN_LABELS.keys())
    x       = np.arange(len(domains))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, lang in enumerate(LANGUAGES):
        scores = []
        for domain in domains:
            items = [r for r in results if r["domain"] == domain]
            score = sum(r[f"correct_{lang}"] for r in items) / len(items) * 100
            scores.append(score)

        bars = ax.bar(x + i * width, scores, width,
                      label=LANG_LABELS[lang],
                      color=COLORS[lang],
                      alpha=0.85,
                      edgecolor="white",
                      linewidth=0.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{score:.0f}%",
                    ha="center", va="bottom", fontsize=9, color="#333333")

    # SSL global annoté
    g = summary["global"]
    ax.annotate(
        f"SSL global : EN−FR = {g['ssl_en_fr']:+.1f}pt  |  EN−AR = {g['ssl_en_ar']:+.1f}pt",
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", fontsize=9.5,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCCCCC")
    )

    ax.set_xticks(x + width)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains], fontsize=11)
    ax.set_ylabel("Taux de bonnes réponses (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Scores par langue et par domaine — Benchmark MMLU",
                 fontsize=13, fontweight="normal", pad=14)
    ax.legend(frameon=False, fontsize=10)
    ax.axhline(y=50, color="#CCCCCC", linestyle="--", linewidth=0.8, label="_nolegend_")

    plt.tight_layout()
    plt.savefig("barplot_scores.png", bbox_inches="tight")
    plt.close()
    print("  ✓ barplot_scores.png")


# ---------------------------------------------------------------------------
# Graphique 2 — Heatmap des réponses correctes
# ---------------------------------------------------------------------------

def plot_heatmap(results: list[dict]):
    items   = [r["item_id"] for r in results]
    data    = {
        "Anglais (EN)":  [r["correct_en"] for r in results],
        "Français (FR)": [r["correct_fr"] for r in results],
        "Arabe (AR)":    [r["correct_ar"] for r in results],
    }
    df = pd.DataFrame(data, index=items).T

    # Couleur de fond par domaine
    domain_colors = {
        "us_history":       "#FFF3E0",
        "moral_philosophy": "#F3E5F5",
        "mathematics":      "#E8F5E9"
    }
    domain_map = {r["item_id"]: r["domain"] for r in results}

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        df,
        ax=ax,
        cmap=["#FFCCCC", "#88C98A"],
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="#EEEEEE",
        cbar=False,
        annot=False
    )

    # Légende manuelle
    patch_correct   = mpatches.Patch(color="#88C98A", label="Réponse correcte")
    patch_incorrect = mpatches.Patch(color="#FFCCCC", label="Réponse incorrecte")
    ax.legend(handles=[patch_correct, patch_incorrect],
              loc="upper right", bbox_to_anchor=(1.0, -0.12),
              ncol=2, frameon=False, fontsize=9)

    # Séparateurs de domaines
    domain_sizes = {"us_history": 10, "moral_philosophy": 10, "mathematics": 10}
    cumulative = 0
    for domain, size in domain_sizes.items():
        cumulative += size
        if cumulative < len(items):
            ax.axvline(x=cumulative, color="white", linewidth=2.5)

    # Labels de domaine en dessous
    domain_positions = [5, 15, 25]
    domain_names     = ["Histoire américaine", "Philosophie morale", "Mathématiques"]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(domain_positions)
    ax2.set_xticklabels(domain_names, fontsize=9)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 30))
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax.set_title("Réponses correctes par item et par langue", fontsize=13,
                 fontweight="normal", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig("heatmap_robustness.png", bbox_inches="tight")
    plt.close()
    print("  ✓ heatmap_robustness.png")


# ---------------------------------------------------------------------------
# Graphique 3 — Distribution de l'indice R
# ---------------------------------------------------------------------------

def plot_robustness_distribution(results: list[dict]):
    r_values = [r["robustness_R"] for r in results]
    domains  = [r["domain"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Histogramme global ---
    ax = axes[0]
    bins = [0, 0.5, 0.8, 1.01]
    colors_hist = ["#E57373", "#FFB74D", "#81C784"]
    labels_hist = ["Inversé (R < 0.5)", "Instable (0.5–0.8)", "Stable (R ≥ 0.8)"]

    for i in range(len(bins) - 1):
        subset = [v for v in r_values if bins[i] <= v < bins[i+1]]
        ax.bar(
            x=[(bins[i] + bins[i+1]) / 2],
            height=len(subset),
            width=(bins[i+1] - bins[i]) * 0.85,
            color=colors_hist[i],
            alpha=0.85,
            label=f"{labels_hist[i]} : {len(subset)} items",
            edgecolor="white"
        )
        if subset:
            ax.text(
                (bins[i] + bins[i+1]) / 2,
                len(subset) + 0.3,
                str(len(subset)),
                ha="center", va="bottom", fontsize=12, fontweight="bold"
            )

    ax.set_xlim(-0.05, 1.1)
    ax.set_xlabel("Indice de robustesse R", fontsize=11)
    ax.set_ylabel("Nombre d'items", fontsize=11)
    ax.set_title("Distribution de l'indice R\n(tous domaines)", fontsize=12, fontweight="normal")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.axvline(x=0.5, color="#AAAAAA", linestyle="--", linewidth=0.8)
    ax.axvline(x=0.8, color="#AAAAAA", linestyle="--", linewidth=0.8)

    # --- Boxplot par domaine ---
    ax2 = axes[1]
    domain_data  = defaultdict(list)
    for r_val, domain in zip(r_values, domains):
        domain_data[domain].append(r_val)

    domain_order  = ["us_history", "moral_philosophy", "mathematics"]
    domain_names  = [DOMAIN_LABELS[d] for d in domain_order]
    plot_data     = [domain_data[d] for d in domain_order]
    domain_colors_box = ["#F18F01", "#A23B72", "#2E86AB"]

    bp = ax2.boxplot(plot_data, patch_artist=True, widths=0.45,
                     medianprops=dict(color="white", linewidth=2))

    for patch, color in zip(bp["boxes"], domain_colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Points individuels
    for i, (data, color) in enumerate(zip(plot_data, domain_colors_box), 1):
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        ax2.scatter(np.full(len(data), i) + jitter, data,
                    color=color, alpha=0.6, s=30, zorder=3)

    ax2.set_xticks(range(1, len(domain_names) + 1))
    ax2.set_xticklabels(domain_names, fontsize=9)
    ax2.set_ylabel("Indice de robustesse R", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_title("Distribution de R par domaine", fontsize=12, fontweight="normal")
    ax2.axhline(y=0.5, color="#AAAAAA", linestyle="--", linewidth=0.8)
    ax2.axhline(y=0.8, color="#AAAAAA", linestyle="--", linewidth=0.8)

    plt.suptitle("Indice de robustesse linguistique — MMLU Stress-Test",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("histogram_R.png", bbox_inches="tight")
    plt.close()
    print("  ✓ histogram_R.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for fname in [RESULTS_FILE, SUMMARY_FILE]:
        if not Path(fname).exists():
            print(f"[ERREUR] Fichier introuvable : {fname}")
            print("Lance d'abord : python run_experiment.py && python metrics.py")
            return

    print("\nGénération des visualisations...")
    results = load_results()
    summary = load_summary()

    plot_scores_by_domain(results, summary)
    plot_heatmap(results)
    plot_robustness_distribution(results)

    print("\n  3 graphiques générés dans le dossier courant.")
    print("  → barplot_scores.png")
    print("  → heatmap_robustness.png")
    print("  → histogram_R.png\n")


if __name__ == "__main__":
    main()
