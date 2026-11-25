"""Render report figures from results/*.json into results/figures/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
})


def _labels(ax, bars, fmt="{:.2f}"):
    """Write each bar's value just above it."""
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK)


def grouped(ax, groups, series, values, ymax=1.0, fmt="{:.2f}"):
    """Grouped bar chart: one cluster per group, one coloured bar per series.
    `values[series][group]` gives each bar's height."""
    n = len(series)
    w = 0.8 / n
    for j, name in enumerate(series):
        # offset each series so the bars of a group are centred on its tick
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(groups))]
        bars = ax.bar(xs, [values[name][g] for g in groups], width=w - 0.03,
                      color=SERIES[j], label=name, edgecolor=SURFACE, linewidth=1)
        _labels(ax, bars, fmt)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, ymax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=n, fontsize=9)


def main():
    """Draw every figure whose results file exists."""
    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    p = RES / "normalization.json"
    if p.exists():
        d = json.loads(p.read_text())["ablation"]
        names = [k for k in d if k != "full w/o synonyms"]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bars = ax.bar(range(len(names)), [d[k]["all"] for k in names], color=SERIES[0], width=0.6)
        _labels(ax, bars)
        ax.set_xticks(range(len(names)), names, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy (160 labelled items)")
        ax.grid(axis="x", visible=False)
        ax.set_title("D1 · Ingredient normalization, cumulative ablation", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / "normalization_ablation.png", dpi=180)
        made.append("normalization_ablation.png")

    p = RES / "retrieval_noisy.json"
    if p.exists():
        d = json.loads(p.read_text())
        modes = ["bm25", "dense", "hybrid", "hybrid_rerank"]
        conds = {"clean queries": "clean", "noisy, raw": "noisy_raw", "noisy, normalized": "noisy_normalized"}
        vals = {c: {m: d[k][m]["MRR"] for m in modes} for c, k in conds.items()}
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        grouped(ax, modes, list(conds), vals, ymax=1.08)
        ax.set_xticks(range(len(modes)), ["BM25", "Dense", "Hybrid (RRF)", "Hybrid + rerank"])
        ax.set_ylabel("MRR (hidden recipe)")
        ax.set_title("D2 · Retrieval MRR by mode and query condition (500 queries)", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / "retrieval_mrr.png", dpi=180)
        made.append("retrieval_mrr.png")

    for p in sorted(RES.glob("generation_*.json")):
        d = json.loads(p.read_text())
        metrics = {"Ingredient grounding": "ingredient_grounding", "Step grounding": "step_grounding"}
        vals = {"PantryRAG": {m: d["rag"][k] or 0 for m, k in metrics.items()},
                "Plain LLM": {m: d["plain"][k] or 0 for m, k in metrics.items()}}
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [2, 1]})
        grouped(axes[0], list(metrics), ["PantryRAG", "Plain LLM"], vals, ymax=1.12)
        axes[0].set_ylabel("Share grounded")
        um = {"PantryRAG": {"Unflagged missing / answer": d["rag"]["unflagged_missing"]},
              "Plain LLM": {"Unflagged missing / answer": d["plain"]["unflagged_missing"]}}
        top = max(v["Unflagged missing / answer"] for v in um.values()) * 1.3 + 0.1
        grouped(axes[1], ["Unflagged missing / answer"], ["PantryRAG", "Plain LLM"], um, ymax=top)
        axes[1].get_legend().remove()
        fig.suptitle(f"D3 · Faithfulness, RAG vs plain LLM ({d['backend']}, n={d['rag']['n']})",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        name = p.stem.replace("generation_", "faithfulness_") + ".png"
        fig.savefig(FIG / name, dpi=180)
        made.append(name)
    print("figures:", made)


if __name__ == "__main__":
    main()
