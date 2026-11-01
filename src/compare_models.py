"""
compare_models.py
-----------------
Produces a side-by-side comparison of the Logistic Regression and
Random Forest models trained on the UNSW-NB15 dataset.

Uses previously recorded evaluation metrics -- does NOT load, retrain,
or re-evaluate either model.

Outputs
-------
  reports/model_comparison.csv          -- all metrics in tabular form
  reports/model_comparison.png          -- grouped bar chart (5 metrics)
  reports/model_comparison_fp_fn.png    -- grouped bar chart (FP vs FN)
  reports/model_comparison_report.txt   -- human-readable written report

No training scripts, saved models, or datasets are modified.

Usage:
    python src/compare_models.py
"""

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

CSV_PATH   = REPORTS_DIR / "model_comparison.csv"
PNG_PATH   = REPORTS_DIR / "model_comparison.png"
FP_FN_PATH = REPORTS_DIR / "model_comparison_fp_fn.png"
TXT_PATH   = REPORTS_DIR / "model_comparison_report.txt"


# ---------------------------------------------------------------------------
# Previously recorded metrics  (hardcoded -- no model loading)
# ---------------------------------------------------------------------------
# Each dictionary maps metric name -> value for one model.
# These values come directly from the training run logs and are the single
# source of truth for every output this script produces.

LR_METRICS = {
    "model":             "Logistic Regression",
    # --- Overall ---
    "accuracy":          0.8358,
    # --- Normal class (label=0) ---
    "normal_precision":  0.8943,
    "normal_recall":     0.7195,
    "normal_f1":         0.7975,
    # --- Attack class (label=1) ---
    "attack_precision":  0.8026,
    "attack_recall":     0.9306,
    "attack_f1":         0.8619,
    # --- Threshold-independent ---
    "roc_auc":           0.9560,
    # --- Confusion matrix counts ---
    "tn":                26_622,
    "fp":                10_378,
    "fn":                 3_145,
    "tp":                42_187,
    # --- Practical costs ---
    "training_time_sec": 14.9,
    "model_size_kb":     8.3,
}

RF_METRICS = {
    "model":             "Random Forest",
    "accuracy":          0.8988,
    "normal_precision":  0.9627,
    "normal_recall":     0.8062,
    "normal_f1":         0.8775,
    "attack_precision":  0.8603,
    "attack_recall":     0.9745,
    "attack_f1":         0.9139,
    "roc_auc":           0.9843,
    "tn":                29_828,
    "fp":                 7_172,
    "fn":                 1_156,
    "tp":                44_176,
    "training_time_sec": 150.5,
    "model_size_kb":     103_100,  # 103.1 MB expressed in KB for a uniform unit
}

# Ordered list of (dict_key, chart_label) pairs used in the bar chart
CHART_METRICS = [
    ("accuracy",         "Accuracy"),
    ("attack_precision", "Attack Precision"),
    ("attack_recall",    "Attack Recall"),
    ("attack_f1",        "Attack F1"),
    ("roc_auc",          "ROC-AUC"),
]


# ---------------------------------------------------------------------------
# Step 1 -- CSV
# ---------------------------------------------------------------------------

def save_csv() -> None:
    """
    Write both models' metrics to a tidy CSV file.

    Each row is one model.  Columns cover every recorded metric plus
    confusion-matrix counts, training time, and model size.
    """
    df = pd.DataFrame([LR_METRICS, RF_METRICS])

    # Rename internal keys to friendlier column headers
    df = df.rename(columns={
        "model":             "Model",
        "accuracy":          "Accuracy",
        "normal_precision":  "Normal Precision",
        "normal_recall":     "Normal Recall",
        "normal_f1":         "Normal F1",
        "attack_precision":  "Attack Precision",
        "attack_recall":     "Attack Recall",
        "attack_f1":         "Attack F1",
        "roc_auc":           "ROC-AUC",
        "tn":                "TN (True Negative)",
        "fp":                "FP (False Positive)",
        "fn":                "FN (False Negative)",
        "tp":                "TP (True Positive)",
        "training_time_sec": "Training Time (s)",
        "model_size_kb":     "Model Size (KB)",
    })

    df.to_csv(CSV_PATH, index=False)
    print(f"    Saved: {CSV_PATH}")


# ---------------------------------------------------------------------------
# Step 2 -- Grouped bar chart (5 performance metrics)
# ---------------------------------------------------------------------------

def save_bar_chart() -> None:
    """
    Create a grouped bar chart comparing the five key performance metrics.

    Both models are shown side-by-side for each metric.
    Y-axis is 0% to 100% (with a small top margin).
    Value labels are printed on top of every bar.
    """
    labels    = [label for _, label in CHART_METRICS]
    lr_values = [LR_METRICS[key] * 100 for key, _ in CHART_METRICS]
    rf_values = [RF_METRICS[key] * 100 for key, _ in CHART_METRICS]

    n     = len(labels)
    x     = range(n)    # integer x positions, one per metric group
    width = 0.35        # width of each individual bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # One bar per model per metric, offset left/right of the group centre
    bars_lr = ax.bar(
        [i - width / 2 for i in x], lr_values,
        width=width,
        label="Logistic Regression",
        color="#4C72B0",
        edgecolor="white",
    )
    bars_rf = ax.bar(
        [i + width / 2 for i in x], rf_values,
        width=width,
        label="Random Forest",
        color="#55A868",
        edgecolor="white",
    )

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(
        "Model Comparison: Logistic Regression vs Random Forest\n"
        "UNSW-NB15 Network Intrusion Detection Dataset",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 108)   # headroom above 100% for value labels
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Print the numeric value just above each bar
    for bar in list(bars_lr) + list(bars_rf):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"    Saved: {PNG_PATH}")


# ---------------------------------------------------------------------------
# Step 3 -- Grouped bar chart (False Positives and False Negatives)
# ---------------------------------------------------------------------------

def save_fp_fn_chart() -> None:
    """
    Create a grouped bar chart comparing raw FP and FN counts.

    Why raw counts matter:
        Percentage metrics can obscure the operational impact.
        Knowing "RF generates 3,206 fewer false alarms per 82k connections"
        is more actionable for a security team than "precision improved
        5.8 percentage points".
    """
    categories = ["False Positives\n(false alarms)", "False Negatives\n(missed attacks)"]
    lr_counts  = [LR_METRICS["fp"], LR_METRICS["fn"]]
    rf_counts  = [RF_METRICS["fp"], RF_METRICS["fn"]]

    n     = len(categories)
    x     = range(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_lr = ax.bar(
        [i - width / 2 for i in x], lr_counts,
        width=width,
        label="Logistic Regression",
        color="#C44E52",
        edgecolor="white",
    )
    bars_rf = ax.bar(
        [i + width / 2 for i in x], rf_counts,
        width=width,
        label="Random Forest",
        color="#DD8452",
        edgecolor="white",
    )

    ax.set_ylabel("Count (out of 82,332 test rows)", fontsize=11)
    ax.set_title(
        "Error Comparison: False Positives and False Negatives\n"
        "UNSW-NB15 Network Intrusion Detection Dataset",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    # Format y-axis tick labels as comma-separated integers (e.g. 10,378)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    for bar in list(bars_lr) + list(bars_rf):
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(FP_FN_PATH, dpi=150)
    plt.close(fig)
    print(f"    Saved: {FP_FN_PATH}")


# ---------------------------------------------------------------------------
# Step 4 -- Written report
# ---------------------------------------------------------------------------

def _diff(rf_val: float, lr_val: float, pct: bool = False) -> str:
    """
    Return a signed difference string.

    pct=True  -- values are 0-1 proportions; express change in percentage
                 points (pp).  e.g. 0.9306 -> 0.9745 = "+4.39 pp"
    pct=False -- values are raw counts; format with commas.
                 e.g. 3145 -> 1156 = "-1,989"
    """
    delta = rf_val - lr_val
    if pct:
        return f"{delta * 100:+.2f} pp"
    return f"{delta:+,.0f}"


def save_report() -> None:
    """
    Write a structured, human-readable text report to
    reports/model_comparison_report.txt.
    """
    lines = []
    W = 70   # separator width

    def sep(title: str = "") -> None:
        """Append a section separator, optionally with a centred title."""
        if title:
            pad   = max(0, W - len(title) - 2)
            left  = pad // 2
            right = pad - left
            lines.append(f"\n{'=' * left} {title} {'=' * right}")
        else:
            lines.append("=" * W)

    # ---- Header ------------------------------------------------------------
    sep()
    lines.append("  MODEL COMPARISON REPORT")
    lines.append("  UNSW-NB15 Network Intrusion Detection Dataset")
    lines.append("  Generated by : src/compare_models.py")
    lines.append("  Test set size: 82,332 rows  "
                 "(37,000 normal  /  45,332 attack)")
    sep()

    # ---- Per-model summaries -----------------------------------------------
    for m in (LR_METRICS, RF_METRICS):
        sep(m["model"].upper())
        lines.append(
            f"\n  Accuracy         : {m['accuracy']:.4f}"
            f"  ({m['accuracy'] * 100:.2f}%)"
        )
        lines.append(
            "  -- How many of all 82,332 predictions were correct?"
        )

        lines.append("\n  Attack class (label = 1)  -- intrusion traffic")
        lines.append(
            f"    Precision : {m['attack_precision']:.4f}"
            f"  ({m['attack_precision'] * 100:.2f}%)"
        )
        lines.append(
            "    -- Of flagged attacks, what fraction were real attacks?"
        )
        lines.append(
            f"    Recall    : {m['attack_recall']:.4f}"
            f"  ({m['attack_recall'] * 100:.2f}%)"
        )
        lines.append(
            "    -- Of all real attacks, what fraction did we catch?"
        )
        lines.append(
            f"    F1-score  : {m['attack_f1']:.4f}"
            f"  ({m['attack_f1'] * 100:.2f}%)"
        )

        lines.append("\n  Normal class (label = 0)  -- legitimate traffic")
        lines.append(
            f"    Precision : {m['normal_precision']:.4f}"
            f"  ({m['normal_precision'] * 100:.2f}%)"
        )
        lines.append(
            f"    Recall    : {m['normal_recall']:.4f}"
            f"  ({m['normal_recall'] * 100:.2f}%)"
        )
        lines.append(
            f"    F1-score  : {m['normal_f1']:.4f}"
            f"  ({m['normal_f1'] * 100:.2f}%)"
        )

        lines.append(
            f"\n  ROC-AUC          : {m['roc_auc']:.4f}"
            f"  ({m['roc_auc'] * 100:.2f}%)"
        )
        lines.append(
            "  -- Threshold-independent separability score.\n"
            "     0.50 = random guessing  |  1.00 = perfect separation"
        )

        lines.append(
            "\n  Confusion matrix (out of 82,332 test rows):\n"
            f"    TN (correct normal)  : {m['tn']:>7,}  "
            "-- normal correctly identified\n"
            f"    FP (false alarms)    : {m['fp']:>7,}  "
            "-- normal wrongly flagged as attack\n"
            f"    FN (missed attacks)  : {m['fn']:>7,}  "
            "-- real attack not detected  <-- minimise this\n"
            f"    TP (correct attacks) : {m['tp']:>7,}  "
            "-- attack correctly identified"
        )

        sz = m["model_size_kb"]
        size_str = (f"{sz:.1f} KB" if sz < 1024 else f"{sz / 1024:.1f} MB")
        lines.append(
            f"\n  Training time    : {m['training_time_sec']:.1f} seconds"
        )
        lines.append(f"  Saved model size : {size_str}")

    # ---- Side-by-side diff table -------------------------------------------
    sep("PERFORMANCE DIFFERENCES  (Random Forest minus Logistic Regression)")
    lines.append("")
    lines.append(
        f"  {'Metric':<28} {'LR':>10} {'RF':>10} {'Change':>12}"
    )
    lines.append(f"  {'-' * 64}")

    pct_rows = [
        ("Accuracy",         "accuracy"),
        ("Attack Precision",  "attack_precision"),
        ("Attack Recall",     "attack_recall"),
        ("Attack F1-score",   "attack_f1"),
        ("Normal Precision",  "normal_precision"),
        ("Normal Recall",     "normal_recall"),
        ("Normal F1-score",   "normal_f1"),
        ("ROC-AUC",           "roc_auc"),
    ]
    for label, key in pct_rows:
        lr_v = LR_METRICS[key]
        rf_v = RF_METRICS[key]
        lines.append(
            f"  {label:<28}"
            f" {lr_v * 100:>9.2f}%"
            f" {rf_v * 100:>9.2f}%"
            f" {_diff(rf_v, lr_v, pct=True):>12}"
        )

    lines.append("")
    lines.append(
        f"  {'Metric':<28} {'LR':>10} {'RF':>10} {'Change':>12}"
    )
    lines.append(f"  {'-' * 64}")

    count_rows = [
        ("False Positives (FP)", "fp"),
        ("False Negatives (FN)", "fn"),
        ("True Positives  (TP)", "tp"),
        ("True Negatives  (TN)", "tn"),
    ]
    for label, key in count_rows:
        lr_v = LR_METRICS[key]
        rf_v = RF_METRICS[key]
        lines.append(
            f"  {label:<28}"
            f" {lr_v:>10,}"
            f" {rf_v:>10,}"
            f" {_diff(rf_v, lr_v, pct=False):>12}"
        )

    # ---- Trade-offs --------------------------------------------------------
    sep("TRADE-OFFS")

    time_ratio = (RF_METRICS["training_time_sec"]
                  / LR_METRICS["training_time_sec"])
    size_ratio = RF_METRICS["model_size_kb"] / LR_METRICS["model_size_kb"]

    lines.append(f"""
  Training time
    Logistic Regression : {LR_METRICS['training_time_sec']:.1f} seconds
    Random Forest       : {RF_METRICS['training_time_sec']:.1f} seconds
    Ratio               : RF takes ~{time_ratio:.0f}x longer to train

  Model size on disk
    Logistic Regression : {LR_METRICS['model_size_kb']:.1f} KB
    Random Forest       : {RF_METRICS['model_size_kb'] / 1024:.1f} MB
    Ratio               : RF is ~{size_ratio:,.0f}x larger on disk

  Interpretation
    Training time matters if you retrain regularly (e.g. weekly on new
    traffic data).  RF at ~2.5 minutes per run is still practical for
    periodic retraining.

    Model size matters for deployment on embedded or memory-constrained
    devices.  On a standard server or workstation it has no impact.

    For inference (predicting on live traffic), both models are fast:
    batch prediction takes milliseconds regardless of model size.
""")

    # ---- Recommendation ----------------------------------------------------
    sep("CAUTIOUS RECOMMENDATION")

    lr_fn = LR_METRICS["fn"]
    rf_fn = RF_METRICS["fn"]
    lr_fp = LR_METRICS["fp"]
    rf_fp = RF_METRICS["fp"]

    lines.append(f"""
  Based solely on the recorded test-set results:

  For a security-focused deployment where MISSING an attack is the
  worst outcome, Random Forest is the stronger choice:
    - Attack recall   : {RF_METRICS['attack_recall'] * 100:.2f}% vs {LR_METRICS['attack_recall'] * 100:.2f}%
    - Missed attacks  : {rf_fn:,} vs {lr_fn:,}  ({lr_fn - rf_fn:,} fewer per 82,332 tested)

  For a deployment where FALSE ALARMS are costly (e.g. analyst
  capacity is limited), Random Forest also wins on precision:
    - Attack precision: {RF_METRICS['attack_precision'] * 100:.2f}% vs {LR_METRICS['attack_precision'] * 100:.2f}%
    - False alarms    : {rf_fp:,} vs {lr_fp:,}  ({lr_fp - rf_fp:,} fewer per 82,332 tested)

  IMPORTANT caveats
    1. These results are from a single held-out test split.
       Cross-validation would give a more reliable estimate of how
       well each model generalises to unseen data.
    2. UNSW-NB15 is a synthetic lab capture.  Performance on real
       enterprise traffic may differ significantly.
    3. No hyperparameter search was performed for either model.
       A tuned Logistic Regression or a tuned Random Forest may show
       different relative performance.
    4. Training times and model sizes were measured on one specific
       machine and will vary by hardware.

  Primary recommendation : Random Forest    (better on every metric)
  Budget-constrained alt : Logistic Regression  (fast, 8 KB, 83.6% acc)
""")

    sep()
    lines.append("END OF REPORT")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TXT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"    Saved: {TXT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 62)
    print("  MODEL COMPARISON")
    print("=" * 62)
    print(f"\nOutput directory : {REPORTS_DIR}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("\nStep 1/4 -- Saving CSV ...")
        save_csv()

        print("\nStep 2/4 -- Saving performance bar chart ...")
        save_bar_chart()

        print("\nStep 3/4 -- Saving FP/FN bar chart ...")
        save_fp_fn_chart()

        print("\nStep 4/4 -- Saving written report ...")
        save_report()

    except Exception:
        # Print the full Python traceback (file, line, error message) so the
        # problem is easy to diagnose, then exit with a non-zero code.
        print("\n[ERROR] Something went wrong while generating the comparison.")
        print("        Full details below:")
        print("-" * 62)
        traceback.print_exc()
        print("-" * 62)
        sys.exit(1)

    print("\n" + "=" * 62)
    print("  DONE -- all four output files written")
    print("=" * 62)
    print(f"\n  {CSV_PATH.name:<48} metrics table")
    print(f"  {PNG_PATH.name:<48} performance bar chart")
    print(f"  {FP_FN_PATH.name:<48} FP / FN bar chart")
    print(f"  {TXT_PATH.name:<48} written report")


if __name__ == "__main__":
    main()
