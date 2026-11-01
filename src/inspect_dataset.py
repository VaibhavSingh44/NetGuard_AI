"""
inspect_dataset.py
------------------
Loads the UNSW-NB15 training and testing CSV files, performs a thorough
inspection, and saves a readable report to reports/dataset_inspection.txt.

The original CSV files are NEVER modified.
Run from the project root:
    python src/inspect_dataset.py
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root, wherever you call the script from)
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent   # …/IBM BOB Project/
TRAIN_PATH   = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH    = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
REPORT_DIR   = ROOT / "reports"
REPORT_PATH  = REPORT_DIR / "dataset_inspection.txt"

LABEL_COL    = "label"
CAT_COL      = "attack_cat"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "", width: int = 70) -> str:
    """Return a section separator line, optionally with a centred title."""
    if title:
        pad = max(0, width - len(title) - 2)
        left = pad // 2
        right = pad - left
        return f"\n{'=' * left} {title} {'=' * right}"
    return "=" * width


def _df_info_string(df: pd.DataFrame) -> str:
    """Capture df.info() output as a string (it prints to stdout by default)."""
    buf = io.StringIO()
    df.info(buf=buf, memory_usage="deep")
    return buf.getvalue()


def section(lines: list, title: str = "") -> None:
    """Append a separator + lines to the shared output list."""
    lines.append(_sep(title))


# ---------------------------------------------------------------------------
# Core inspection
# ---------------------------------------------------------------------------

def inspect(df: pd.DataFrame, name: str, lines: list) -> None:
    """
    Run all checks on a single DataFrame and append human-readable text
    to `lines`.  `name` is a short label like 'TRAINING' or 'TESTING'.
    """

    # --- 2. Shape, columns, dtypes, memory ----------------------------------
    section(lines, f"{name} SET — SHAPE & COLUMNS")

    lines.append(f"\nShape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    lines.append("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        lines.append(f"  {i:>3}. {col}")

    lines.append("\nData types (value counts):")
    for dtype, count in df.dtypes.value_counts().items():
        lines.append(f"  {str(dtype):<15} {count} column(s)")

    lines.append("\nPer-column dtypes:")
    for col in df.columns:
        lines.append(f"  {col:<30} {df[col].dtype}")

    section(lines, f"{name} SET — MEMORY USAGE")
    mem_bytes = df.memory_usage(deep=True).sum()
    lines.append(f"\nTotal in-memory size : {mem_bytes / 1_048_576:.2f} MB")
    lines.append("\nPer-column memory (bytes):")
    for col, mem in df.memory_usage(deep=True).items():
        if col == "Index":
            lines.append(f"  {'<index>':<30} {mem:>12,} bytes")
        else:
            lines.append(f"  {col:<30} {mem:>12,} bytes")

    # --- 3. Label column -----------------------------------------------------
    section(lines, f"{name} SET — LABEL DISTRIBUTION")
    if LABEL_COL not in df.columns:
        lines.append(f"\n[WARNING] Column '{LABEL_COL}' not found.")
    else:
        counts = df[LABEL_COL].value_counts().sort_index()
        total  = len(df)
        lines.append(f"\nColumn : '{LABEL_COL}'")
        lines.append(f"{'Value':<10} {'Count':>10} {'Percentage':>12}")
        lines.append("-" * 36)
        for val, cnt in counts.items():
            lines.append(f"{str(val):<10} {cnt:>10,} {cnt / total * 100:>11.2f}%")
        lines.append(f"{'TOTAL':<10} {total:>10,} {'100.00%':>12}")

    # --- 4. Attack-category column -------------------------------------------
    section(lines, f"{name} SET — ATTACK CATEGORIES")
    if CAT_COL not in df.columns:
        lines.append(f"\n[INFO] Column '{CAT_COL}' not found — skipping.")
    else:
        raw_counts = df[CAT_COL].fillna("").astype(str).str.strip()
        # Treat empty / whitespace-only strings as "Normal"
        raw_counts = raw_counts.replace("", "Normal")
        counts = raw_counts.value_counts().sort_values(ascending=False)
        total  = len(df)
        # lines.append(f"\nColumn : '{CAT_COL}'  ({counts.nunique()} unique categories)") # counts.nunique() counts the number of distinct frequency values, not the number of attack categories.
        # if three attack categories have counts of 100, 200, and 300, this expression returns 3 by coincidence. If two categories both have a count of 100, it would undercount the categories.
        lines.append(f"\nColumn : '{CAT_COL}'  ({len(counts)} unique categories)")
        lines.append(f"{'Category':<30} {'Count':>10} {'Percentage':>12}")
        lines.append("-" * 56)
        for cat, cnt in counts.items():
            lines.append(f"{str(cat):<30} {cnt:>10,} {cnt / total * 100:>11.2f}%")

    # --- 5a. Missing values ---------------------------------------------------
    section(lines, f"{name} SET — MISSING VALUES")
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if cols_with_nulls.empty:
        lines.append("\nNo missing (NaN) values found.")
    else:
        lines.append(f"\n{len(cols_with_nulls)} column(s) have missing values:")
        lines.append(f"{'Column':<30} {'NaN Count':>12} {'% of rows':>12}")
        lines.append("-" * 58)
        for col, cnt in cols_with_nulls.items():
            lines.append(
                f"{col:<30} {cnt:>12,} {cnt / len(df) * 100:>11.2f}%"
            )

    # --- 5b. Infinity check ---------------------------------------------------
    section(lines, f"{name} SET — INFINITY CHECK")
    num_df = df.select_dtypes(include=[np.number])
    pos_inf = (num_df == np.inf).sum()
    neg_inf = (num_df == -np.inf).sum()
    inf_cols = pos_inf[pos_inf > 0].index.union(neg_inf[neg_inf > 0].index)
    if inf_cols.empty:
        lines.append("\nNo positive or negative infinity values found.")
    else:
        lines.append(f"\n{len(inf_cols)} column(s) contain infinity values:")
        lines.append(f"{'Column':<30} {'+Inf':>10} {'-Inf':>10}")
        lines.append("-" * 54)
        for col in inf_cols:
            lines.append(f"{col:<30} {pos_inf[col]:>10,} {neg_inf[col]:>10,}")

    # --- 5c. Duplicate column names -------------------------------------------
    section(lines, f"{name} SET — DUPLICATE COLUMN NAMES")
    seen, dupes = set(), []
    for col in df.columns:
        if col in seen:
            dupes.append(col)
        seen.add(col)
    if dupes:
        lines.append(f"\n{len(dupes)} duplicate column name(s) found: {dupes}")
    else:
        lines.append("\nNo duplicate column names found.")

    # --- 5d. Duplicate rows ---------------------------------------------------
    section(lines, f"{name} SET — DUPLICATE ROWS")
    dup_count = df.duplicated().sum()
    lines.append(
        f"\nDuplicate rows : {dup_count:,}  "
        f"({dup_count / len(df) * 100:.2f}% of total)"
    )

    # --- 7. Five sample rows -------------------------------------------------
    section(lines, f"{name} SET — 5 SAMPLE ROWS (first 5)")
    lines.append("")
    # Use to_string with a fixed width so it fits in the report
    sample_str = df.head(5).to_string(max_cols=15, max_colwidth=20)
    lines.append(sample_str)


# ---------------------------------------------------------------------------
# Column-compatibility check (requirement 6)
# ---------------------------------------------------------------------------

def check_compatibility(train: pd.DataFrame, test: pd.DataFrame,
                         lines: list) -> None:
    section(lines, "COLUMN COMPATIBILITY — TRAIN vs TEST")

    train_cols = set(train.columns)
    test_cols  = set(test.columns)

    only_train = sorted(train_cols - test_cols)
    only_test  = sorted(test_cols  - train_cols)
    common     = sorted(train_cols & test_cols)

    lines.append(f"\nTraining columns : {len(train.columns)}")
    lines.append(f"Testing  columns : {len(test.columns)}")
    lines.append(f"Common   columns : {len(common)}")

    if only_train:
        lines.append(f"\nColumns ONLY in training ({len(only_train)}): {only_train}")
    else:
        lines.append("\nNo columns exclusive to training.")

    if only_test:
        lines.append(f"\nColumns ONLY in testing ({len(only_test)}): {only_test}")
    else:
        lines.append("\nNo columns exclusive to testing.")

    # dtype mismatches among shared columns
    mismatches = [
        (col, train[col].dtype, test[col].dtype)
        for col in common
        if train[col].dtype != test[col].dtype
    ]
    if mismatches:
        lines.append(f"\nDtype mismatches in {len(mismatches)} shared column(s):")
        lines.append(f"  {'Column':<30} {'Train dtype':<15} {'Test dtype':<15}")
        lines.append("  " + "-" * 62)
        for col, td, tt in mismatches:
            lines.append(f"  {col:<30} {str(td):<15} {str(tt):<15}")
    else:
        lines.append("\nAll shared columns have matching dtypes.")

    if not only_train and not only_test and not mismatches:
        lines.append("\n[OK] Datasets are fully compatible.")
    else:
        lines.append("\n[!!] Compatibility issues detected (see above).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # --- 1. Load CSV files ---------------------------------------------------
    for path, label in [(TRAIN_PATH, "training"), (TEST_PATH, "testing")]:
        if not path.exists():
            print(
                f"[ERROR] Cannot find the {label} CSV file.\n"
                f"        Expected location: {path}\n"
                f"        Make sure the file is placed there and re-run the script."
            )
            sys.exit(1)

    print("Loading training set ...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"  -> {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")

    print("Loading testing set ...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  -> {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    # Collect all output lines in memory first, then print + save together
    lines: list = []
    lines.append("UNSW-NB15 DATASET INSPECTION REPORT")
    lines.append(f"Generated by : src/inspect_dataset.py")
    lines.append(f"Training file: {TRAIN_PATH}")
    lines.append(f"Testing  file: {TEST_PATH}")

    # Inspect each set
    inspect(train_df, "TRAINING", lines)
    inspect(test_df,  "TESTING",  lines)

    # Column compatibility
    check_compatibility(train_df, test_df, lines)

    lines.append(_sep())
    lines.append("END OF REPORT")

    # --- 8. Save report -------------------------------------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    # Also print to terminal
    print("\n" + report_text)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
