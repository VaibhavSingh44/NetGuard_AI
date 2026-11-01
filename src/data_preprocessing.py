"""
data_preprocessing.py
---------------------
Loads the UNSW-NB15 training and testing CSVs, validates the data,
builds a reusable scikit-learn preprocessing Pipeline, and prints a
feature summary.

The original CSV files are NEVER modified.

Usage:
    python src/data_preprocessing.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths  (relative to the project root, wherever you call the script from)
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
TRAIN_PATH  = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH   = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

# ---------------------------------------------------------------------------
# Column roles  (single source of truth — edit only here if columns change)
# ---------------------------------------------------------------------------
LABEL_COL = "label"

# These three columns are excluded from the feature matrix X entirely.
#   id         — row identifier, carries no predictive signal
#   attack_cat — encodes attack type; keeping it would be "label leakage"
#                (the model would cheat by reading the answer from the input)
#   label      — the target we want to predict, not an input feature
DROP_COLS = ["id", "attack_cat", "label"]

# These three columns are plain text (e.g. "tcp", "udp", "ftp") and must be
# converted to numbers via OneHotEncoder before a model can use them.
CAT_COLS = ["proto", "service", "state"]

# All remaining columns (derived automatically below) are treated as numbers
# and passed through StandardScaler.


# ---------------------------------------------------------------------------
# Helper: missing-file guard
# ---------------------------------------------------------------------------

def _check_file(path: Path, label: str) -> None:
    """Exit with a clear, actionable message when a required CSV is missing."""
    if not path.exists():
        print(
            f"\n[ERROR] The {label} CSV file was not found."
            f"\n        Expected location: {path}"
            f"\n        Place the file there and re-run the script."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: label validation
# ---------------------------------------------------------------------------

# def validate_label(df: pd.DataFrame, name: str) -> None:
#     """
#     Confirm the 'label' column exists and contains only the values 0 and 1.

#     Why this matters:
#         If the column has typos, extra classes, or unexpected nulls, every
#         metric computed later would be silently wrong.  Better to catch it
#         here than hours into training.
#     """
#     if LABEL_COL not in df.columns:
#         print(f"\n[ERROR] '{LABEL_COL}' column not found in the {name} set.")
#         sys.exit(1)

#     actual_values = set(int(v) for v in df[LABEL_COL].dropna().unique())
#     unexpected    = actual_values - {0, 1}

#     if unexpected:
#         print(
#             f"\n[ERROR] '{LABEL_COL}' in the {name} set contains unexpected "
#             f"values: {sorted(unexpected)}"
#             f"\n        Expected only 0 (normal) and 1 (attack)."
#         )
#         sys.exit(1)

#     print(f"  [{name}] label OK  -- unique values: {sorted(actual_values)}")

def validate_label(df: pd.DataFrame, name: str) -> None:
    if LABEL_COL not in df.columns:
        raise ValueError(f"'{LABEL_COL}' column not found in the {name} set.")

    if df[LABEL_COL].isna().any():
        raise ValueError(f"'{LABEL_COL}' contains missing values in the {name} set.")

    actual_values = set(df[LABEL_COL].unique())

    if not actual_values.issubset({0, 1}):
        raise ValueError(f"Unexpected label values in {name}: {actual_values}. "
            "Expected only 0 (normal) and 1 (attack).")

    print(f"  [{name}] label OK -- unique values: {sorted(actual_values)}")


# ---------------------------------------------------------------------------
# Helper: derive numerical column list automatically
# ---------------------------------------------------------------------------

def derive_num_cols(all_columns: pd.Index) -> list:
    """
    Return the list of numerical feature columns by removing every column
    that belongs to DROP_COLS or CAT_COLS.

    Doing this automatically means:
      - You never have to maintain a long hardcoded list of 39 columns.
      - If the dataset ever adds a new numerical column, it is included
        without any edits to this file.
    """
    excluded = set(DROP_COLS) | set(CAT_COLS)
    return [col for col in all_columns if col not in excluded]


# ---------------------------------------------------------------------------
# Helper: feature compatibility check
# ---------------------------------------------------------------------------

def validate_feature_compatibility(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list,
) -> None:
    """
    Verify that every feature column selected from the training set also
    exists in the testing set, and vice versa.

    Why this matters:
        If a column is present in training but missing in testing, the
        ColumnTransformer will raise a cryptic KeyError during prediction.
        Catching it here gives a human-readable message instead.
    """
    feature_cols     = CAT_COLS + num_cols
    missing_in_test  = [c for c in feature_cols if c not in test_df.columns]
    missing_in_train = [c for c in feature_cols if c not in train_df.columns]

    if missing_in_test:
        print(
            f"\n[ERROR] Feature column(s) present in training but missing "
            f"in testing: {missing_in_test}"
        )
        sys.exit(1)

    if missing_in_train:
        print(
            f"\n[ERROR] Feature column(s) present in testing but missing "
            f"in training: {missing_in_train}"
        )
        sys.exit(1)

    print("  Feature columns are compatible between training and testing sets.")


# ---------------------------------------------------------------------------
# Core: build the reusable preprocessing Pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(num_cols: list) -> Pipeline:
    """
    Build and return an UNFITTED scikit-learn Pipeline.

    What "unfitted" means:
        The pipeline does not yet know the mean/std of any number column,
        nor the full set of categories in proto/service/state.  It learns
        those statistics only when you call  pipeline.fit(X_train)  or
        pipeline.fit_transform(X_train)  later.  This is the crucial
        guarantee that test-set information never leaks into training.

    Pipeline structure
    ------------------
    Pipeline
    └── ColumnTransformer
        ├── "cat" branch  OneHotEncoder  --> proto, service, state
        └── "num" branch  StandardScaler --> all 39 numerical columns

    Parameters
    ----------
    num_cols : list[str]
        Column names to treat as numerical features.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready to be fitted — no data has been touched yet.
    """

    # ------------------------------------------------------------------
    # Categorical branch
    # ------------------------------------------------------------------
    # OneHotEncoder converts a text column into a set of binary (0/1)
    # columns — one per unique category.
    #
    # Example:  proto = "tcp"  →  [1, 0, 0]   (if categories are tcp/udp/icmp)
    #           proto = "udp"  →  [0, 1, 0]
    #
    # handle_unknown="ignore":
    #   If the test set contains a protocol the model never saw during
    #   training (e.g. a rare new protocol), produce an all-zero vector
    #   instead of crashing.  Essential for real-world robustness.
    #
    # sparse_output=False:
    #   Return a regular dense NumPy array instead of a sparse matrix.
    #   Easier to inspect and compatible with all downstream estimators.
    # ------------------------------------------------------------------
    # categorical_transformer = OneHotEncoder(
    #     handle_unknown="ignore",
    #     sparse_output=False,
    # )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ------------------------------------------------------------------
    # Numerical branch
    # ------------------------------------------------------------------
    # StandardScaler rescales each column to have mean = 0, std = 1.
    #
    # Why it matters:
    #   "sbytes" (source bytes) can reach millions while "dur" (duration)
    #   is often less than 1.  Without scaling, distance-based models and
    #   gradient-based optimisers treat sbytes as if it is a million times
    #   more important than dur — which it isn't.
    #
    # Formula applied to each value x in a column:
    #   x_scaled = (x - column_mean) / column_std
    #
    # Wrapped in its own Pipeline so it is easy to swap out later
    # (e.g. replace StandardScaler with RobustScaler) in one place.
    # ------------------------------------------------------------------
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    # ------------------------------------------------------------------
    # ColumnTransformer — route each column to the right branch
    # ------------------------------------------------------------------
    # Each tuple is ("name", transformer, list_of_columns).
    # remainder="drop" silently discards any column not listed — this is
    # a safety net in case X accidentally still contains id/attack_cat.
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CAT_COLS),
            ("num", numerical_transformer,   num_cols),
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------------
    # Wrap in a top-level Pipeline
    # ------------------------------------------------------------------
    # A Pipeline chains steps so the whole preprocessing + model (added
    # in the next task) behaves as a single object:
    #
    #   pipeline.fit(X_train, y_train)
    #       → fits preprocessor on X_train
    #       → fits model on the transformed output
    #
    #   pipeline.predict(X_test)
    #       → transforms X_test using the ALREADY-FITTED preprocessor
    #       → returns model predictions
    #
    # The model slot is left as a comment placeholder for the next step.
    # ------------------------------------------------------------------
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        # ("model", YourClassifierHere())   <-- added in the training script
    ])

    return full_pipeline


# ---------------------------------------------------------------------------
# Public API — called by this script and imported by future scripts
# ---------------------------------------------------------------------------

def load_and_prepare():
    """
    Orchestrate the full preprocessing setup.

    Steps
    -----
    1.  Confirm both CSV files exist.
    2.  Load them into DataFrames (read-only).
    3.  Validate the label column in each set.
    4.  Derive feature column lists automatically.
    5.  Validate feature compatibility between sets.
    6.  Split each DataFrame into X (features) and y (labels).
    7.  Print a human-readable feature summary.
    8.  Build and return the unfitted preprocessing pipeline.

    Returns
    -------
    X_train : pd.DataFrame   -- feature matrix for training (175,341 x 42)
    X_test  : pd.DataFrame   -- feature matrix for testing  ( 82,332 x 42)
    y_train : pd.Series      -- labels for training
    y_test  : pd.Series      -- labels for testing
    pipeline: Pipeline       -- unfitted preprocessing pipeline
    num_cols: list[str]      -- names of the 39 numerical feature columns
    """

    # ---- 1. File existence check ----------------------------------------
    _check_file(TRAIN_PATH, "training")
    _check_file(TEST_PATH,  "testing")

    # ---- 2. Load (read-only) --------------------------------------------
    print("Loading training set ...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"  -> {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")

    print("Loading testing set ...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  -> {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    # ---- 3. Validate labels ---------------------------------------------
    print("\nValidating label column ...")
    validate_label(train_df, "TRAINING")
    validate_label(test_df,  "TESTING")

    # ---- 4. Derive numerical columns ------------------------------------
    num_cols = derive_num_cols(train_df.columns)

    # ---- 5. Validate cross-set feature compatibility --------------------
    print("\nValidating feature column compatibility ...")
    validate_feature_compatibility(train_df, test_df, num_cols)

    # ---- 6. Split X and y  (official predefined split is preserved) -----
    #   X: every column except id, attack_cat, and label
    #   y: label column only
    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[LABEL_COL]

    X_test  = test_df.drop(columns=DROP_COLS)
    y_test  = test_df[LABEL_COL]

    # ---- 7. Feature summary ---------------------------------------------
    print(f"\nFeature summary")
    print(f"  Categorical features : {len(CAT_COLS)}  -> {CAT_COLS}")
    print(f"  Numerical features   : {len(num_cols)}")
    print(f"  Total input features : {len(CAT_COLS) + len(num_cols)}")
    print(f"\n  Numerical columns ({len(num_cols)}):")
    for col in num_cols:
        print(f"    {col}")

    print(f"\n  X_train shape : {X_train.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  y_train shape : {y_train.shape}  "
          f"(0={( y_train==0).sum():,}  1={(y_train==1).sum():,})")
    print(f"  y_test  shape : {y_test.shape}  "
          f"(0={(y_test ==0).sum():,}  1={(y_test ==1).sum():,})")

    # ---- 8. Build the unfitted pipeline ---------------------------------
    pipeline = build_preprocessing_pipeline(num_cols)

    print("\nPipeline built (unfitted):")
    print(pipeline)

    print(
        "\n[OK] Preprocessing setup complete."
        "\n     The pipeline is ready to be fitted on training data."
        "\n     No model has been trained.  No CSV files were modified."
    )

    return X_train, X_test, y_train, y_test, pipeline, num_cols


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_and_prepare()
