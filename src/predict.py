"""
predict.py
----------
Batch prediction interface for the UNSW-NB15 Random Forest intrusion
detector.

Loads the saved fitted pipeline (preprocessing + model) and runs it on
one of two input sources chosen by the MODE variable near the top:

  MODE = "sample"   -- takes N random rows from the official test CSV,
                       strips the label column, predicts, then compares
                       to the real labels so you can see where the model
                       is right and wrong.

  MODE = "manual"   -- predicts on the small hand-crafted records defined
                       in MANUAL_RECORDS, requiring no CSV file at all.

This is a BATCH PREDICTION PROTOTYPE.  It does not monitor live network
traffic and does not claim production readiness.

The model file, CSV files, and all other project files are NEVER modified.

Usage:
    python src/predict.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths  (derived from this file's location -- no hard-coded absolute paths)
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest_intrusion_detector.joblib"
TEST_CSV   = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

# ---------------------------------------------------------------------------
# Feature contract  (must match data_preprocessing.py exactly)
# ---------------------------------------------------------------------------
# These three columns must never be passed as model inputs.
DROP_COLS = ["id", "attack_cat", "label"]

# Categorical columns -- converted to one-hot vectors by the saved pipeline.
CAT_COLS = ["proto", "service", "state"]

# Numerical columns -- scaled by the saved pipeline.
# This exact list was extracted from the fitted ColumnTransformer inside
# models/random_forest_intrusion_detector.joblib.  Do not edit unless
# you retrain the model with a different column set.
NUM_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
    "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]

# All 42 feature columns the model expects as input
FEATURE_COLS = CAT_COLS + NUM_COLS

# Decision boundary: >= this probability -> predicted Attack
# 0.5 means "majority vote" -- more than half of the 150 trees said attack.
ATTACK_PROB_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Prediction mode
# ---------------------------------------------------------------------------
# Change MODE to switch between the two input sources described above.
MODE        = "sample"
SAMPLE_N    = 10    # number of rows to draw from the test CSV
SAMPLE_SEED = 99    # fixed seed so the same 10 rows are drawn every run

# ---------------------------------------------------------------------------
# Manual records  (used only when MODE = "manual")
# ---------------------------------------------------------------------------
# Each dictionary represents one network connection.
# Provide values for all 42 feature columns.
# Any column you omit will be filled with a safe default (0 for numbers,
# "-" for text) and a warning will be printed.
#
# Record 0: short UDP burst with no reply -- typical of normal internal traffic
# Record 1: TCP session with high byte count and mismatched TTL values --
#           pattern seen in several attack categories
MANUAL_RECORDS = [
    {
        "proto": "udp",  "service": "-",   "state": "INT",
        "dur": 0.000011, "spkts": 2,       "dpkts": 0,
        "sbytes": 496,   "dbytes": 0,      "rate": 90909.09,
        "sttl": 254,     "dttl": 0,        "sload": 361163584.0,
        "dload": 0.0,    "sloss": 0,       "dloss": 0,
        "sinpkt": 0.011, "dinpkt": 0.0,    "sjit": 0.0,
        "djit": 0.0,     "swin": 0,        "stcpb": 0,
        "dtcpb": 0,      "dwin": 0,        "tcprtt": 0.0,
        "synack": 0.0,   "ackdat": 0.0,    "smean": 248,
        "dmean": 0,      "trans_depth": 0, "response_body_len": 0,
        "ct_srv_src": 1, "ct_state_ttl": 2, "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 2,   "is_ftp_login": 0,
        "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
        "ct_srv_dst": 2, "is_sm_ips_ports": 0,
    },
    {
        "proto": "tcp",  "service": "-",   "state": "FIN",
        "dur": 1.623129, "spkts": 8,       "dpkts": 16,
        "sbytes": 521,   "dbytes": 14763,  "rate": 14.8,
        "sttl": 62,      "dttl": 252,      "sload": 2566.7,
        "dload": 72724.8,"sloss": 0,       "dloss": 0,
        "sinpkt": 232.73,"dinpkt": 92.82,  "sjit": 62.1,
        "djit": 97.3,    "swin": 255,      "stcpb": 1972024704,
        "dtcpb": 1453850624, "dwin": 255,  "tcprtt": 0.049,
        "synack": 0.025, "ackdat": 0.024,  "smean": 65,
        "dmean": 923,    "trans_depth": 0, "response_body_len": 0,
        "ct_srv_src": 2, "ct_state_ttl": 1, "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 6,   "is_ftp_login": 0,
        "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 2,
        "ct_srv_dst": 6, "is_sm_ips_ports": 0,
    },
]


# ---------------------------------------------------------------------------
# Helper: load the saved pipeline
# ---------------------------------------------------------------------------

def load_pipeline():
    """
    Load the fitted Random Forest pipeline from disk using joblib.

    The file contains:
      - A fitted ColumnTransformer (OneHotEncoder + StandardScaler)
        that already knows the training categories and mean/std values.
      - A fitted RandomForestClassifier with 150 trained decision trees.

    No retraining occurs here -- all weights are already stored in the file.
    The file is ~103 MB so loading takes around 10-20 seconds.
    """
    if not MODEL_PATH.exists():
        print(
            f"\n[ERROR] Model file not found."
            f"\n        Expected: {MODEL_PATH}"
            f"\n        Run src/train_random_forest.py first to create it."
        )
        sys.exit(1)

    print(f"Loading pipeline from: {MODEL_PATH.name}")
    print("(The model file is ~103 MB -- this may take 10-20 seconds.  "
          "Please wait ...)")
    pipeline = joblib.load(MODEL_PATH)
    print("Pipeline loaded successfully.\n")
    return pipeline


# ---------------------------------------------------------------------------
# Helper: validate and clean the input DataFrame
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a raw DataFrame for prediction by the saved pipeline.

    Steps performed:
      1. Remove any DROP_COLS columns (id, attack_cat, label) if present.
         This is a safety net -- the label must never be passed as a feature.
      2. Drop any unrecognised extra columns with a notice.
      3. Fill any missing feature columns with safe defaults and warn.
      4. Return the DataFrame with exactly the 42 columns in the right order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input rows that may contain extra or missing columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with exactly FEATURE_COLS as columns.
    """

    # Step 1: remove label/id/attack_cat so they cannot leak into the model
    leaked = [c for c in DROP_COLS if c in df.columns]
    if leaked:
        print(f"[NOTICE] Removing non-feature column(s) from input: {leaked}")
        df = df.drop(columns=leaked)

    # Step 2: drop columns the model was never trained on
    extra = [c for c in df.columns if c not in FEATURE_COLS]
    if extra:
        print(f"[NOTICE] Dropping unrecognised column(s): {extra}")
        df = df.drop(columns=extra)

    # Step 3: fill missing feature columns with safe defaults
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] {len(missing)} feature column(s) missing from input.")
        print("          They will be filled with default values:")
        for col in missing:
            if col in CAT_COLS:
                df[col] = "-"
                print(f"  {col:<30} -> '-'  (categorical default)")
            else:
                df[col] = 0
                print(f"  {col:<30} ->  0   (numerical default)")
        print()

    # Step 4: reorder to the exact sequence the ColumnTransformer expects
    return df[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Helper: print prediction results
# ---------------------------------------------------------------------------

def print_results(
    df_input: pd.DataFrame,
    predictions: list,
    probabilities: list,
    true_labels: list = None,
) -> None:
    """
    Print one result block per input record, with probability explanation.

    Parameters
    ----------
    df_input      : cleaned 42-column input DataFrame
    predictions   : list of int (0 or 1), one per row
    probabilities : list of float (0.0-1.0) attack probability, one per row
    true_labels   : list of actual labels, or None if not available
    """
    sep = "=" * 62
    n   = len(predictions)

    print(f"\n{sep}")
    print(f"  PREDICTION RESULTS  ({n} record{'s' if n != 1 else ''})")
    print(sep)

    correct            = 0
    total_with_labels  = 0

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label_str = "Normal (0)" if pred == 0 else "Attack (1)"
        pct        = prob * 100

        print(f"\n  Record {i + 1}")
        print(f"  {'-' * 40}")

        # Show the categorical features -- most human-readable for diagnosis
        for col in CAT_COLS:
            print(f"    {col:<12}: {df_input.iloc[i][col]}")

        print(f"\n  Prediction      : {label_str}")

        # ---- Probability display -------------------------------------------
        # predict_proba returns the fraction of the 150 trees that voted
        # "attack".  A value of 0.93 means 140 of 150 trees said "attack".
        # This is a model estimate, not a definitive security verdict.
        print(f"  Attack prob.    : {pct:.1f}%  "
              f"(decision threshold: {ATTACK_PROB_THRESHOLD * 100:.0f}%)")

        if prob >= ATTACK_PROB_THRESHOLD:
            print(f"  Interpretation  : {pct:.1f}% of the model's decision")
            print( "                    trees voted 'attack'.")
            print( "                    -> Classified as ATTACK.")
        else:
            print(f"  Interpretation  : Only {pct:.1f}% of the model's decision")
            print( "                    trees voted 'attack'.")
            print( "                    -> Classified as NORMAL.")

        print( "  NOTE: This is a model estimate based on network flow")
        print( "        features only.  It is NOT a definitive security")
        print( "        verdict and should not be used as one.")

        # ---- Ground-truth comparison (sample mode only) --------------------
        if true_labels is not None:
            actual     = int(true_labels[i])
            actual_str = "Normal (0)" if actual == 0 else "Attack (1)"
            match      = (pred == actual)
            status     = "CORRECT" if match else "WRONG"
            print(f"\n  Actual label    : {actual_str}  [{status}]")
            if match:
                correct += 1
            total_with_labels += 1

    # ---- Summary -----------------------------------------------------------
    if true_labels is not None and total_with_labels > 0:
        print(f"\n{sep}")
        acc = correct / total_with_labels * 100
        print(f"  SAMPLE ACCURACY : {correct}/{total_with_labels} correct  "
              f"({acc:.1f}%)")
        print(f"  (Expected ~90% based on full test-set evaluation)")
        print(sep)

    print(f"\n[DONE] {n} prediction(s) complete.")
    print("       This script is a batch prediction prototype.")
    print("       It does not monitor live network traffic.")


# ---------------------------------------------------------------------------
# Mode: draw a random sample from the test CSV
# ---------------------------------------------------------------------------

def run_sample_mode(pipeline) -> None:
    """
    Pull SAMPLE_N rows from the official test CSV, strip the label before
    predicting, then compare predictions to real labels afterwards.

    Why there is no data leakage here
    ----------------------------------
    The model was already trained and saved -- nothing is fitted here.
    The real 'label' values are copied out BEFORE validate_input() strips
    them, so the label never enters pipeline.predict() as a feature.
    The real labels are only used AFTER prediction to measure accuracy.
    """
    if not TEST_CSV.exists():
        print(
            f"\n[ERROR] Test CSV not found."
            f"\n        Expected: {TEST_CSV}"
            f"\n        This file is required for sample mode."
        )
        sys.exit(1)

    print(f"Sample mode  : {SAMPLE_N} random rows from the test CSV")
    print(f"Random seed  : {SAMPLE_SEED}  (same rows drawn every run)")
    print(f"Source file  : {TEST_CSV.name}\n")

    df_raw    = pd.read_csv(TEST_CSV)
    df_sample = df_raw.sample(n=SAMPLE_N, random_state=SAMPLE_SEED)

    # Save real labels BEFORE cleaning strips them -- prevents any leakage
    true_labels = df_sample["label"].tolist()

    # validate_input() removes 'label', 'id', and 'attack_cat'
    df_input = validate_input(df_sample.copy())

    print(f"Input shape after validation : {df_input.shape}  "
          f"(expected {SAMPLE_N} x {len(FEATURE_COLS)})")

    predictions   = pipeline.predict(df_input).tolist()
    probabilities = pipeline.predict_proba(df_input)[:, 1].tolist()

    print_results(df_input, predictions, probabilities, true_labels)


# ---------------------------------------------------------------------------
# Mode: predict on hand-crafted manual records
# ---------------------------------------------------------------------------

def run_manual_mode(pipeline) -> None:
    """
    Predict on the records defined in MANUAL_RECORDS at the top of this file.
    No CSV is needed.  Real labels are not available so accuracy cannot be
    measured here.
    """
    n = len(MANUAL_RECORDS)
    print(f"Manual mode  : {n} hand-crafted record{'s' if n != 1 else ''}")
    print("               (No CSV file needed in this mode)\n")

    df_raw   = pd.DataFrame(MANUAL_RECORDS)
    df_input = validate_input(df_raw.copy())

    print(f"Input shape after validation : {df_input.shape}  "
          f"(expected {n} x {len(FEATURE_COLS)})")

    predictions   = pipeline.predict(df_input).tolist()
    probabilities = pipeline.predict_proba(df_input)[:, 1].tolist()

    # No true_labels available for hand-crafted records
    print_results(df_input, predictions, probabilities, true_labels=None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 62)
    print("  UNSW-NB15 INTRUSION DETECTOR -- BATCH PREDICTION")
    print("=" * 62)
    print(f"\nModel : {MODEL_PATH.name}")
    print(f"Mode  : {MODE}")
    print()

    pipeline = load_pipeline()

    if MODE == "sample":
        run_sample_mode(pipeline)
    elif MODE == "manual":
        run_manual_mode(pipeline)
    else:
        print(
            f"\n[ERROR] Unknown MODE '{MODE}'."
            f"\n        Set MODE to 'sample' or 'manual' near the top of"
            f"\n        src/predict.py and re-run."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
