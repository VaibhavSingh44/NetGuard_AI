"""
app.py
------
Streamlit web application for the UNSW-NB15 Random Forest intrusion detector.

Users upload a CSV of network-flow records.  The app:
  1. Validates column compatibility against the trained pipeline's feature schema.
  2. Strips id, attack_cat, and label before prediction (preserving labels
     for optional evaluation if they are present).
  3. Displays per-record predictions, estimated attack probabilities, and
     a downloadable results CSV.
  4. Optionally evaluates predictions against actual labels when the upload
     contains a 'label' column.

This is a PROTOTYPE for offline / batch use.
It does not perform live packet capture or real-time network monitoring.
Model probabilities are estimated class probabilities, not guaranteed
confidence levels.  The application is not production-ready.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths  (resolved from this file's location — no hard-coded absolute paths)
# ---------------------------------------------------------------------------
# __file__ is  app/app.py
# .parent        → app/
# .parent.parent → project root  (F:\IBM BOB Project)
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest_intrusion_detector.joblib"

# ---------------------------------------------------------------------------
# Feature schema  (sourced directly from the fitted pipeline)
# ---------------------------------------------------------------------------
# These constants mirror src/predict.py exactly so the web app and the
# batch script always agree on what the model expects.

# Columns that must NEVER enter the model as input features.
DROP_COLS = ["id", "attack_cat", "label"]

# Text columns — encoded as one-hot vectors by the saved pipeline.
CAT_COLS = ["proto", "service", "state"]

# Numerical columns — standardised by the saved pipeline.
# This list was extracted from the fitted ColumnTransformer inside
# models/random_forest_intrusion_detector.joblib.
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

# All 42 columns the model expects as input (order matters for ColumnTransformer)
FEATURE_COLS = CAT_COLS + NUM_COLS


# ---------------------------------------------------------------------------
# Model loading  (cached so the 103 MB file is only read once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Random Forest pipeline (~103 MB) …")
def load_pipeline():
    """
    Load the fitted pipeline from disk exactly once per Streamlit session.

    st.cache_resource stores the object in memory so navigating the page or
    re-uploading a file does not reload the 103 MB model every time.

    Returns None when the model file does not exist so the caller can show
    a friendly error instead of a raw exception.
    """
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_upload(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str], list[str]]:
    """
    Inspect an uploaded DataFrame and return cleaned model inputs.

    Validation order
    ----------------
    1. Reject empty uploads.
    2. Extract actual labels (if present) BEFORE any column is dropped.
    3. Remove DROP_COLS and report which ones were removed.
    4. Report and remove unrecognised extra columns.
    5. Reject uploads with missing required feature columns.
    6. Reject uploads containing NaN or infinite values in numerical columns.
    7. Reorder columns to the exact sequence the ColumnTransformer expects.

    Returns
    -------
    X : pd.DataFrame or None
        42-column feature matrix ready for pipeline.predict(), or None on error.
    y_true : pd.Series or None
        Actual labels preserved from the upload, or None if absent/invalid.
    warnings : list[str]
        Non-fatal notices shown to the user as st.warning().
    errors : list[str]
        Fatal problems shown as st.error() — prediction cannot proceed.
    """
    warnings: list[str] = []
    errors:   list[str] = []

    # 1. Empty upload guard
    if df.empty:
        errors.append("The uploaded file contains no data rows.")
        return None, None, warnings, errors

    # 2. Extract actual labels BEFORE anything is dropped.
    #    Labels are preserved separately and only used after predictions are
    #    generated — they are never passed to pipeline.predict().
    y_true: pd.Series | None = None
    if "label" in df.columns:
        raw_vals = set(df["label"].dropna().unique())
        invalid  = raw_vals - {0, 1}
        if invalid:
            warnings.append(
                f"The 'label' column contains unexpected value(s): "
                f"{sorted(invalid)}.  Expected only 0 (Normal) and 1 (Attack).  "
                f"The evaluation section will be skipped."
            )
            # y_true stays None — evaluation section will not render
        else:
            y_true = df["label"].copy()

    # 3. Remove DROP_COLS and report which ones were stripped
    removed = [c for c in DROP_COLS if c in df.columns]
    if removed:
        warnings.append(
            f"The following non-feature column(s) were removed before "
            f"prediction to prevent data leakage: {removed}"
        )
        df = df.drop(columns=removed)

    # 4. Report and drop unrecognised columns
    extra = [c for c in df.columns if c not in FEATURE_COLS]
    if extra:
        warnings.append(
            f"The following column(s) are not part of the trained feature "
            f"set and were dropped: {extra}"
        )
        df = df.drop(columns=extra)

    # 5. Reject uploads with missing required feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        cat_missing = [c for c in missing if c in CAT_COLS]
        num_missing = [c for c in missing if c in NUM_COLS]
        msg = (
            f"**{len(missing)} required feature column(s) are absent** from "
            f"the upload and cannot be inferred:\n\n"
        )
        if cat_missing:
            msg += f"- Categorical: `{'`, `'.join(cat_missing)}`\n"
        if num_missing:
            msg += f"- Numerical: `{'`, `'.join(num_missing)}`\n"
        msg += (
            "\nPredictions on incomplete inputs would be unreliable.  "
            "Please supply all 42 feature columns."
        )
        errors.append(msg)
        return None, y_true, warnings, errors

    # 6. Reject NaN and infinite values in numerical columns
    X = df[FEATURE_COLS].copy()
    num_data = X[NUM_COLS]

    nan_cols = [c for c in NUM_COLS if num_data[c].isna().any()]
    if nan_cols:
        errors.append(
            f"Missing (NaN) values found in numerical column(s): "
            f"`{'`, `'.join(nan_cols)}`.  "
            f"Please clean the data before uploading."
        )
        return None, y_true, warnings, errors

    inf_cols = [c for c in NUM_COLS if np.isinf(num_data[c]).any()]
    if inf_cols:
        errors.append(
            f"Infinite values found in numerical column(s): "
            f"`{'`, `'.join(inf_cols)}`.  "
            f"Please clean the data before uploading."
        )
        return None, y_true, warnings, errors

    # 7. Reorder to the exact column sequence the ColumnTransformer expects
    return X[FEATURE_COLS], y_true, warnings, errors


# ---------------------------------------------------------------------------
# Results builder
# ---------------------------------------------------------------------------

def build_results_df(
    df_original: pd.DataFrame,
    predictions:  np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Combine the original upload with the prediction outputs.

    The returned DataFrame contains:
      - All original columns except 'id' and 'attack_cat'
        (the 'label' column is kept if present so the user can
         compare manually — it was never used as model input).
      - predicted_label     : 0 or 1
      - predicted_class     : "Normal" or "Attack"
      - attack_probability  : estimated attack probability (0.0000 – 1.0000)
    """
    # Drop only id and attack_cat; keep label if present for transparency
    out = df_original.drop(
        columns=[c for c in ["id", "attack_cat"] if c in df_original.columns]
    ).copy()
    out["predicted_label"]    = predictions
    out["predicted_class"]    = ["Attack" if p == 1 else "Normal"
                                  for p in predictions]
    out["attack_probability"] = np.round(probabilities, 4)
    return out


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

def main() -> None:

    # ---- Page configuration ------------------------------------------------
    st.set_page_config(
        page_title="Network Intrusion Detector",
        page_icon="🔒",
        layout="wide",
    )

    st.title("🔒 Network Intrusion Detector")
    st.caption(
        "UNSW-NB15 dataset · Random Forest classifier · "
        "Offline batch prediction prototype"
    )

    st.info(
        "**Prototype notice** — This tool performs offline batch prediction "
        "on uploaded network-flow records.  It does **not** monitor live "
        "network traffic.  Predicted probabilities are estimated class "
        "probabilities from the trained model, not guaranteed confidence "
        "levels.  This application is not production-ready.",
        icon="ℹ️",
    )

    # ---- Load model --------------------------------------------------------
    # load_pipeline() is cached; it only reads the file on the first call.
    pipeline = load_pipeline()

    if pipeline is None:
        st.error(
            f"**Model file not found.**\n\n"
            f"Expected location: `{MODEL_PATH}`\n\n"
            f"Run `src/train_random_forest.py` to create it, then "
            f"restart the app."
        )
        st.stop()   # nothing below this line runs when the model is missing

    # Model info panel
    with st.expander("ℹ️ Model information", expanded=False):
        st.markdown(f"""
| Property | Value |
|---|---|
| File | `{MODEL_PATH.name}` |
| Algorithm | RandomForestClassifier — 150 trees |
| Categorical input features | `proto`, `service`, `state` |
| Numerical input features | {len(NUM_COLS)} columns |
| Total input features | {len(FEATURE_COLS)} |
| Preprocessing | OneHotEncoder + StandardScaler (fitted on UNSW-NB15 training data) |
        """)

    st.divider()

    # ---- Sidebar: column reference -----------------------------------------
    with st.sidebar:
        st.header("Required columns")
        st.markdown(
            "Your CSV must contain all **42 feature columns** listed below.  \n"
            "The columns `id`, `attack_cat`, and `label` are **optional** — "
            "they are removed before prediction if present."
        )
        st.markdown("**Categorical (3)**")
        for c in CAT_COLS:
            st.code(c)
        st.markdown(f"**Numerical ({len(NUM_COLS)})**")
        for c in NUM_COLS:
            st.code(c)

    # ---- Step 1: Upload ----------------------------------------------------
    st.header("Step 1 — Upload a CSV file")
    st.markdown(
        "Each row in the file should represent one network-flow record.  "
        "See the sidebar for the full list of required columns."
    )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Each row = one network-flow record.  Recommended max size: 50 MB.",
    )

    if uploaded is None:
        st.info("Waiting for a file upload.  Use the widget above to begin.")
        st.stop()

    # ---- Step 2: Parse and validate ----------------------------------------
    st.header("Step 2 — Validating")

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(
            f"**Could not parse the uploaded file.**\n\n"
            f"The file may be corrupted or not a valid CSV.\n\n"
            f"Detail: `{exc}`"
        )
        st.stop()

    st.success(
        f"File read successfully: "
        f"**{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**"
    )

    with st.expander("Preview: first 5 rows of upload", expanded=False):
        st.dataframe(df_raw.head(), use_container_width=True)

    # Run validation
    X, y_true, val_warnings, val_errors = validate_upload(df_raw.copy())

    for msg in val_warnings:
        st.warning(msg)

    if val_errors:
        for msg in val_errors:
            st.error(msg)
        st.stop()

    st.success(
        f"Validation passed — **{len(X):,} records** ready for prediction "
        f"({len(FEATURE_COLS)} feature columns confirmed)."
    )

    # Let the user know labels were found and will be used for evaluation
    if y_true is not None:
        n_normal_true = int((y_true == 0).sum())
        n_attack_true = int((y_true == 1).sum())
        st.info(
            f"A `label` column was detected: "
            f"**{n_normal_true:,} Normal** / **{n_attack_true:,} Attack**.  "
            f"An evaluation section will appear after predictions are generated."
        )

    # ---- Step 3: Predict ---------------------------------------------------
    st.header("Step 3 — Predictions")

    with st.spinner("Running the Random Forest pipeline …"):
        predictions   = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)[:, 1]

    n_total  = len(predictions)
    n_attack = int((predictions == 1).sum())
    n_normal = int((predictions == 0).sum())

    # Summary tiles
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records",    f"{n_total:,}")
    c2.metric("Predicted Normal", f"{n_normal:,}",
              delta=f"{n_normal / n_total * 100:.1f}%",
              delta_color="off")
    c3.metric("Predicted Attack", f"{n_attack:,}",
              delta=f"{n_attack / n_total * 100:.1f}%",
              delta_color="off")

    # Build results DataFrame (original columns + prediction columns)
    results_df = build_results_df(df_raw, predictions, probabilities)

    # Results table — show a compact view to keep the page readable
    with st.expander("Results table (first 100 rows)", expanded=True):
        display_cols = ["predicted_class", "attack_probability",
                        "proto", "service", "state"]
        # Include label in the display if it was in the upload
        if "label" in results_df.columns:
            display_cols = ["label"] + display_cols
        st.dataframe(
            results_df[display_cols].head(100),
            use_container_width=True,
        )

    # Probability distribution chart
    with st.expander(
        "Estimated attack probability distribution", expanded=False
    ):
        st.markdown(
            "Each bar shows how many records received an estimated attack "
            "probability in that range.  "
            "Records with probability **≥ 0.50** are classified as Attack; "
            "those **< 0.50** as Normal.  "
            "These are model estimates, not certainties."
        )
        hist_vals, hist_edges = np.histogram(probabilities, bins=20,
                                              range=(0.0, 1.0))
        bin_labels = [
            f"{hist_edges[i]:.2f}–{hist_edges[i+1]:.2f}"
            for i in range(len(hist_vals))
        ]
        st.bar_chart(
            pd.DataFrame({"count": hist_vals}, index=bin_labels)
        )

    # Download button
    st.subheader("Download results")
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️  Download predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="intrusion_predictions.csv",
        mime="text/csv",
        help="Downloads all records with predicted_class and attack_probability columns appended.",
    )

    # ---- Step 4: Evaluation (conditional) ----------------------------------
    if y_true is not None:
        st.divider()
        st.header("Step 4 — Evaluation against actual labels")
        st.markdown(
            "Because the upload contained a `label` column, predictions can "
            "be compared to known ground truth.  "
            "**The actual labels were extracted before prediction and were "
            "never passed to the model as input features.**"
        )
        st.warning(
            "These metrics describe performance on this uploaded sample only.  "
            "They do not represent real-world or live-network performance.",
            icon="⚠️",
        )

        y_arr = y_true.values
        acc   = accuracy_score(y_arr, predictions)
        prec  = precision_score(y_arr, predictions, zero_division=0)
        rec   = recall_score(y_arr, predictions,    zero_division=0)
        f1    = f1_score(y_arr, predictions,        zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc  * 100:.2f}%")
        m2.metric("Precision", f"{prec * 100:.2f}%",
                  help="Of all predicted attacks, what fraction were real attacks?")
        m3.metric("Recall",    f"{rec  * 100:.2f}%",
                  help="Of all real attacks, what fraction did the model catch?")
        m4.metric("F1-score",  f"{f1   * 100:.2f}%",
                  help="Harmonic mean of precision and recall.")

        # Confusion matrix
        cm = confusion_matrix(y_arr, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        st.subheader("Confusion matrix")
        st.markdown(
            "Each cell shows the **count** of predictions in that category."
        )
        cm_df = pd.DataFrame(
            {
                "Predicted Normal": [f"{tn:,}", f"{fn:,}"],
                "Predicted Attack": [f"{fp:,}", f"{tp:,}"],
            },
            index=["Actual Normal", "Actual Attack"],
        )
        st.dataframe(cm_df, use_container_width=False)

        st.markdown(
            f"- **TN** — correct normal: **{tn:,}**  \n"
            f"- **FP** — false alarms: **{fp:,}**  \n"
            f"- **FN** — missed attacks: **{fn:,}**  \n"
            f"- **TP** — correct attacks: **{tp:,}**"
        )

        with st.expander("Full classification report", expanded=False):
            report = classification_report(
                y_arr, predictions,
                target_names=["Normal (0)", "Attack (1)"],
                digits=4,
                zero_division=0,
            )
            st.code(report)

    # ---- Footer ------------------------------------------------------------
    st.divider()
    st.caption(
        "UNSW-NB15 Network Intrusion Detection · Random Forest prototype · "
        "Not production-ready · Does not monitor live traffic"
    )


if __name__ == "__main__":
    main()
