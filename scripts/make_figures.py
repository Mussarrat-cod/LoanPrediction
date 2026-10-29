"""
make_figures.py -- Generate publication-quality figures and paper tables
from trained artifacts.

Produces 300-dpi PNGs in artifacts/figures/ and artifacts/paper_tables.md.

Usage:
    python -m scripts.make_figures
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    THRESHOLD_SWEEP,
    DECISION_THRESHOLD,
    LR_PARAMS,
    RF_PARAMS,
    DT_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
    ASSUMED_ANNUAL_INTEREST_RATE,
    LOAN_TERM_UNIT,
    SHAP_BACKGROUND_SIZE,
)
from backend.data_layer import load_dataset, engineer_features
from backend.model_layer import predict_with_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DPI = 300
STYLE_PARAMS = {
    "figure.facecolor": "#0f1219",
    "axes.facecolor": "#171c28",
    "axes.edgecolor": "#2a3448",
    "axes.labelcolor": "#e8ecf4",
    "text.color": "#e8ecf4",
    "xtick.color": "#8b95a8",
    "ytick.color": "#8b95a8",
    "grid.color": "#2a3448",
    "grid.alpha": 0.5,
}
plt.rcParams.update(STYLE_PARAMS)
sns.set_palette(["#4a9eff", "#22c55e", "#8b5cf6", "#f59e0b", "#ef4444", "#06b6d4"])


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    X_test, y_test = joblib.load(ARTIFACTS_DIR / "test_set.joblib")
    X_train, y_train = joblib.load(ARTIFACTS_DIR / "train_set.joblib")
    background = joblib.load(ARTIFACTS_DIR / "background.joblib")

    models = {}
    for name, fname in [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Random Forest", "random_forest.joblib"),
        ("Decision Tree", "decision_tree.joblib"),
    ]:
        path = ARTIFACTS_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)

    best_model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    best_name = metrics.get("best_model", "Unknown")

    # Load full dataset for class distribution
    df = load_dataset()

    # ---- 1. Class distribution ----
    logger.info("1. Class distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["loan_status"].value_counts().sort_index()
    labels = ["Rejected (0)", "Approved (1)"]
    colors = ["#ef4444", "#22c55e"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(val), ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution", fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. Accuracy comparison ----
    logger.info("2. Accuracy comparison")
    model_names_list = [n for n in ["Logistic Regression", "Random Forest", "Decision Tree"] if n in metrics]
    accs = [metrics[n]["accuracy"] * 100 for n in model_names_list]
    f1s = [metrics[n]["f1"] * 100 for n in model_names_list]
    short_names = ["LR", "RF", "DT"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(short_names))
    w = 0.35
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4a9eff", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, f1s, w, label="F1 Score", color="#22c55e", edgecolor="white", linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Accuracy & F1 Comparison", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. Top-10 global feature importance (best model) ----
    logger.info("3. Global feature importance")
    top_imp = metrics.get(best_name, {}).get("top5_importance", [])
    if top_imp:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [d["feature"] for d in top_imp][:10]
        vals = [d["importance"] for d in top_imp][:10]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color="#4a9eff", edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top Feature Importance ({best_name})", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "global_importance.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---- 4. SHAP beeswarm + waterfall (using shap library) ----
    logger.info("4. SHAP plots")
    try:
        import shap
        feature_names = list(background.columns)

        def predict_fn(X):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=feature_names)
            return best_model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, background)

        # Beeswarm on a small sample
        sample = X_test[feature_names].iloc[:50]
        sv = explainer.shap_values(sample.values, nsamples="auto")
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        shap_explanation = shap.Explanation(
            values=np.array(sv),
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0],
            data=sample.values,
            feature_names=feature_names,
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_explanation, show=False, max_display=15)
        fig = plt.gcf()
        fig.patch.set_facecolor("#0f1219")
        fig.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=DPI, bbox_inches="tight", facecolor="#0f1219")
        plt.close("all")

        # Waterfall: one approved, one rejected
        preds = best_model.predict_proba(sample)[:, 1]
        approved_idx = np.argmax(preds)
        rejected_idx = np.argmin(preds)

        for label, idx in [("approved", approved_idx), ("rejected", rejected_idx)]:
            single_exp = shap.Explanation(
                values=np.array(sv[idx]),
                base_values=shap_explanation.base_values if isinstance(shap_explanation.base_values, float) else shap_explanation.base_values[idx],
                data=sample.values[idx],
                feature_names=feature_names,
            )
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(single_exp, show=False, max_display=12)
            fig = plt.gcf()
            fig.patch.set_facecolor("#0f1219")
            fig.savefig(FIGURES_DIR / f"shap_waterfall_{label}.png", dpi=DPI, bbox_inches="tight", facecolor="#0f1219")
            plt.close("all")

        logger.info("SHAP plots saved.")
    except Exception as e:
        logger.warning("SHAP plots failed: %s", e)

    # ---- 5. Confusion matrices ----
    logger.info("5. Confusion matrices")
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        preds, _ = predict_with_threshold(model, X_test, y_test)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontweight="bold", fontsize=11)
    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 6. Calibration curves ----
    logger.info("6. Calibration curves")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_predicted = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_predicted, fraction_pos, "o-", label=name, markersize=4)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves (After Isotonic for LR, DT)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 7. Threshold sweep ----
    logger.info("7. Threshold sweep")
    sweep = metrics.get("threshold_sweep", [])
    if sweep:
        ts = [s["threshold"] for s in sweep]
        prec = [s["precision"] for s in sweep]
        rec = [s["recall"] for s in sweep]
        f1 = [s["f1"] for s in sweep]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ts, prec, "o-", label="Precision", color="#4a9eff", markersize=5)
        ax.plot(ts, rec, "s-", label="Recall", color="#22c55e", markersize=5)
        ax.plot(ts, f1, "^-", label="F1", color="#8b5cf6", markersize=5)
        ax.axvline(DECISION_THRESHOLD, color="#ef4444", linestyle="--", alpha=0.7, label=f"Threshold={DECISION_THRESHOLD}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1 vs Threshold", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---- Paper tables ----
    logger.info("Writing paper_tables.md")
    _write_paper_tables(metrics, df, best_name)

    logger.info("All figures generated in %s", FIGURES_DIR)


def _write_paper_tables(metrics, df, best_name):
    lines = []
    lines.append("# Paper Tables (Auto-generated)\n")

    # Dataset schema
    lines.append("## Table 1: Dataset Schema\n")
    lines.append("| Column | Type | Range |")
    lines.append("|--------|------|-------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        if df[col].dtype in (np.int64, np.float64):
            rng = f"{df[col].min()} - {df[col].max()}"
        else:
            rng = ", ".join(sorted(df[col].unique().astype(str)))
        lines.append(f"| {col} | {dtype} | {rng} |")
    lines.append("")

    # Class distribution
    counts = df["loan_status"].value_counts().sort_index()
    lines.append("## Table 2: Class Distribution\n")
    lines.append("| Class | Count | Percentage |")
    lines.append("|-------|-------|------------|")
    total = counts.sum()
    lines.append(f"| Rejected (0) | {counts.get(0, 0)} | {counts.get(0, 0)/total*100:.1f}% |")
    lines.append(f"| Approved (1) | {counts.get(1, 0)} | {counts.get(1, 0)/total*100:.1f}% |")
    lines.append(f"| **Total** | **{total}** | 100% |")
    lines.append("")

    # Hyperparameters
    lines.append("## Table 3: Hyperparameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Random State | {RANDOM_STATE} |")
    lines.append(f"| Test Size | {TEST_SIZE} |")
    lines.append(f"| Interest Rate | {ASSUMED_ANNUAL_INTEREST_RATE} |")
    lines.append(f"| Loan Term Unit | {LOAN_TERM_UNIT} |")
    lines.append(f"| SHAP Background | {SHAP_BACKGROUND_SIZE} |")
    lines.append(f"| LR | {LR_PARAMS} |")
    lines.append(f"| RF | {RF_PARAMS} |")
    lines.append(f"| DT | {DT_PARAMS} |")
    lines.append("")

    # Model comparison
    lines.append("## Table 4: Model Comparison\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy | CV F1 |")
    lines.append("|-------|----------|-----------|--------|-----|---------|-------------|-------|")
    for name in ["Logistic Regression", "Random Forest", "Decision Tree"]:
        if name not in metrics:
            continue
        m = metrics[name]
        cv_acc = f"{m.get('cv_accuracy_mean', 0)*100:.2f} +/- {m.get('cv_accuracy_std', 0)*100:.2f}" if "cv_accuracy_mean" in m else "-"
        cv_f1 = f"{m.get('cv_f1_mean', 0)*100:.2f} +/- {m.get('cv_f1_std', 0)*100:.2f}" if "cv_f1_mean" in m else "-"
        best_marker = " **(Best)**" if name == best_name else ""
        lines.append(
            f"| {name}{best_marker} | {m['accuracy']*100:.2f}% | {m['precision']*100:.2f}% | "
            f"{m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['roc_auc']*100:.2f}% | {cv_acc} | {cv_f1} |"
        )
    lines.append("")

    # Top-5 importance per model
    lines.append("## Table 5: Top-5 Feature Importance per Model\n")
    for name in ["Logistic Regression", "Random Forest", "Decision Tree"]:
        if name not in metrics:
            continue
        imp = metrics[name].get("top5_importance", [])
        if imp:
            lines.append(f"### {name}\n")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|------|---------|------------|")
            for i, d in enumerate(imp, 1):
                lines.append(f"| {i} | {d['feature']} | {d['importance']:.4f} |")
            lines.append("")

    # Ablation
    abl = metrics.get("RF (no cibil_score) ablation")
    if abl:
        lines.append("## Table 6: Ablation Study (RF without cibil_score)\n")
        lines.append("| Metric | With cibil_score | Without cibil_score |")
        lines.append("|--------|-----------------|---------------------|")
        rf = metrics.get("Random Forest", {})
        for met in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            lines.append(f"| {met} | {rf.get(met, 0)*100:.2f}% | {abl.get(met, 0)*100:.2f}% |")
        lines.append("")

    # Latency
    lat = metrics.get("avg_latency_ms_per_sample", "N/A")
    lines.append(f"## Measured Latency\n\nAverage inference latency: **{lat} ms/sample**\n")

    out_path = ARTIFACTS_DIR / "paper_tables.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
