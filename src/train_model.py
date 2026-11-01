"""
train_model.py
--------------
Trains a Logistic Regression classifier on the UNSW-NB15 dataset using
the reusable preprocessing pipeline from data_preprocessing.py.

Fits only on training data.  Evaluates on the held-out test set.
Saves the fitted pipeline to models/intrusion_detector.joblib.

The original CSV files are NEVER modified.

Usage:
    python src/train_model.py
"""

import sys
import time
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where Python is invoked from.
# sys.path is the list of directories Python searches for modules.
# Inserting src/ at position 0 ensures data_preprocessing.py is found
# before any other file with the same name elsewhere on the system.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import load_and_prepare  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "intrusion_detector.joblib"


# ---------------------------------------------------------------------------
# Evaluation printer
# ---------------------------------------------------------------------------

def print_metrics(y_true, y_pred, y_prob) -> None:
    """
    Print every requested metric with a plain-English explanation.

    Parameters
    ----------
    y_true : array-like
        The correct labels from the test set (0 = normal, 1 = attack).
    y_pred : array-like
        The labels the model predicted for the test set.
    y_prob : array-like
        The model's estimated probability of class 1 (attack) for each row.
        Used only for ROC-AUC — not needed for the other metrics.
    """

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    sep  = "=" * 62

    print(f"\n{sep}")
    print("  INDIVIDUAL METRICS")
    print(sep)

    # -- Accuracy ------------------------------------------------------------
    # The simplest metric: (correct predictions) / (all predictions).
    # Easy to understand but dangerous with imbalanced classes.
    # A model that always shouts "attack!" on this dataset would score ~55%
    # accuracy without learning anything useful.  Read the other metrics too.
    print(f"\n  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(  "              Of ALL predictions, how many were correct?")

    # -- Precision -----------------------------------------------------------
    # Out of every connection the model flagged as an attack, what fraction
    # truly was an attack?
    # Low precision → lots of false alarms → analysts waste time.
    print(f"\n  Precision : {prec:.4f}  ({prec * 100:.2f}%)")
    print(  "              Of predicted attacks, what fraction were real attacks?")

    # -- Recall (Sensitivity) ------------------------------------------------
    # Out of every real attack in the test set, what fraction did the model
    # actually catch?
    # Low recall → real attacks slip through undetected → dangerous.
    # In a security context, recall is usually the most critical metric.
    print(f"\n  Recall    : {rec:.4f}  ({rec * 100:.2f}%)")
    print(  "              Of ALL real attacks, what fraction did we catch?")

    # -- F1-score ------------------------------------------------------------
    # The harmonic mean of precision and recall:
    #   F1 = 2 * (precision * recall) / (precision + recall)
    # It penalises extreme imbalances between the two — a model with perfect
    # recall but zero precision would score F1 = 0, not 0.5.
    print(f"\n  F1-score  : {f1:.4f}  ({f1 * 100:.2f}%)")
    print(  "              Balanced score combining precision and recall.")

    # -- ROC-AUC -------------------------------------------------------------
    # The area under the Receiver Operating Characteristic curve.
    # Intuition: if you pick one random attack and one random normal
    # connection, ROC-AUC is the probability the model scores the attack
    # higher than the normal connection.
    #   0.5 = no better than random guessing
    #   1.0 = perfect separation
    print(f"\n  ROC-AUC   : {roc:.4f}  ({roc * 100:.2f}%)")
    print(  "              Probability model ranks a random attack above a random")
    print(  "              normal connection. 0.5 = random, 1.0 = perfect.")

    # -- Confusion Matrix ----------------------------------------------------
    # A 2x2 table that shows WHERE the model was right and wrong.
    #
    #              | Predicted Normal | Predicted Attack
    # -------------|------------------|------------------
    # Actual Normal|   TN (correct)   |   FP (false alarm)
    # Actual Attack|   FN (missed!)   |   TP (correct)
    #
    #   TN = True  Negative : normal, correctly identified as normal
    #   FP = False Positive : normal, wrongly flagged as attack (false alarm)
    #   FN = False Negative : real attack that the model MISSED  ← minimise this
    #   TP = True  Positive : attack, correctly identified as attack
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"  {'':<28} {'Predicted Normal':>16}  {'Predicted Attack':>16}")
    print(f"  {'-' * 64}")
    print(f"  {'Actual Normal  (label=0)':<28} {tn:>16,}  {fp:>16,}")
    print(f"  {'Actual Attack  (label=1)':<28} {fn:>16,}  {tp:>16,}")
    print(f"  {'-' * 64}")
    print(f"  {'':28}  FP = false alarms      FN = missed attacks")

    # -- Classification Report -----------------------------------------------
    # Per-class breakdown: precision, recall, and F1 for BOTH classes.
    # "support" = number of rows of that class in the test set.
    print(f"\n{sep}")
    print("  CLASSIFICATION REPORT")
    print(sep)
    print()
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal (0)", "Attack (1)"],
            digits=4,
        )
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Full training and evaluation workflow:

      Step 1 — Load data via load_and_prepare() from data_preprocessing.py.
      Step 2 — Add LogisticRegression as the final step in the pipeline.
      Step 3 — Fit the complete pipeline on training data ONLY.
      Step 4 — Evaluate on the held-out test set.
      Step 5 — Save the fitted pipeline to disk with joblib.
    """

    # =========================================================================
    # STEP 1 — LOAD DATA
    # =========================================================================
    # load_and_prepare() does all the heavy lifting:
    #   - reads both CSVs (read-only)
    #   - validates the label column
    #   - splits X (features) and y (labels)
    #   - returns an UNFITTED preprocessing pipeline
    # We discard num_cols here (captured by _ ) — it was only needed inside
    # data_preprocessing.py when building the pipeline.
    print("\n" + "=" * 62)
    print("  STEP 1 — LOAD DATA")
    print("=" * 62)

    X_train, X_test, y_train, y_test, pipeline, _ = load_and_prepare()

    # =========================================================================
    # STEP 2 — ADD LOGISTIC REGRESSION TO THE PIPELINE
    # =========================================================================
    # Up to this point the pipeline contains only the preprocessor.
    # We append the classifier as the second (and final) step.
    #
    # After this the full pipeline is:
    #
    #   Pipeline
    #   ├── "preprocessor"  ColumnTransformer
    #   │   ├── "cat"  OneHotEncoder   (proto, service, state)
    #   │   └── "num"  StandardScaler  (39 numerical columns)
    #   └── "model"  LogisticRegression
    #
    # --- Argument choices explained ------------------------------------------
    #
    #   max_iter=1000
    #       The optimiser adjusts model weights iteratively.  The default 100
    #       is too few for 175k rows x 42+ features — the solver would stop
    #       early.  1000 gives lbfgs enough room to converge cleanly.
    #
    #   solver="lbfgs"
    #       L-BFGS is a quasi-Newton method — it converges in far fewer
    #       iterations than stochastic solvers (e.g. saga) on this feature
    #       size because it uses curvature information to take smarter steps.
    #       It is also the scikit-learn 1.9 recommended default for binary
    #       classification and carries no deprecation warnings.
    #
    #   random_state=42
    #       Seeds the internal random-number generator so every run produces
    #       identical results.  The value 42 is a convention — any integer
    #       works just as well.
    #
    #   class_weight="balanced"
    #       The training set contains approximately 68% attacks and 32% normal connections. 
    #       class_weight="balanced" gives the less frequent normal class greater weight during training. 
    #       This can improve the balance between the classes, but it does not guarantee better recall, precision, or accuracy. 
    #       Those outcomes must be measured.
    #       "balanced" computes per-class weights as:
    #           weight(class) = total_rows / (n_classes * count(class))
    #       so the minority class (normal) is treated as more important,
    #       forcing the model to learn to identify both classes well.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  STEP 2 — BUILD COMPLETE PIPELINE")
    print("=" * 62)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )

    # pipeline.steps is a list of ("name", transformer/estimator) tuples.
    # Appending adds "model" as the new last step.
    pipeline.steps.append(("model", clf))
    pipeline = Pipeline(steps=pipeline.steps)

    print("\nComplete pipeline (unfitted):")
    print(pipeline)

    # =========================================================================
    # STEP 3 — FIT ON TRAINING DATA ONLY
    # =========================================================================
    # pipeline.fit(X_train, y_train) executes the two pipeline steps in order:
    #   a) preprocessor.fit_transform(X_train)
    #      → learns mean/std from training numbers
    #      → learns category lists from training text columns
    #      → transforms X_train into a numeric matrix
    #   b) model.fit(transformed_X_train, y_train)
    #      → finds the best weights for Logistic Regression
    #
    # X_test and y_test are NOT touched here in any way.
    print("\n" + "=" * 62)
    print("  STEP 3 — FIT ON TRAINING DATA")
    print("=" * 62)
    print(f"\nTraining on {len(X_train):,} rows ...")
    print("(Logistic Regression with LBFGS may take some time")
    print("depending on your CPU. Please wait ...)")

    t_start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t_start

    print(f"\nTraining complete in {elapsed:.1f} seconds.")

    # =========================================================================
    # STEP 4 — EVALUATE ON THE HELD-OUT TEST SET
    # =========================================================================
    # pipeline.predict(X_test) runs in two sub-steps automatically:
    #   a) preprocessor.transform(X_test)
    #      → applies the ALREADY-FITTED scaler/encoder to X_test
    #      → uses the mean/std/categories learned from training — nothing new
    #        is learned from the test set
    #   b) model.predict(transformed_X_test)
    #      → returns a 1-D array of 0s and 1s
    #
    # pipeline.predict_proba(X_test)[:, 1]
    #   → returns the probability of class 1 (attack) for each test row
    #   → column index 0 = P(normal), column index 1 = P(attack)
    #   → we only need P(attack) for ROC-AUC
    print("\n" + "=" * 62)
    print("  STEP 4 — EVALUATE ON TEST SET")
    print("=" * 62)
    print(f"\nEvaluating on {len(X_test):,} held-out rows ...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print_metrics(y_test, y_pred, y_prob)

    # =========================================================================
    # STEP 5 — SAVE THE FITTED PIPELINE
    # =========================================================================
    # joblib.dump() serialises the entire fitted pipeline to a single binary
    # file.  "Fitted" means the scaler knows the training mean/std, the
    # encoder knows the training categories, and the model knows its weights.
    #
    # To make predictions later without re-training:
    #   import joblib
    #   pipeline = joblib.load("models/intrusion_detector.joblib")
    #   y_pred   = pipeline.predict(X_new)
    #
    # X_new only needs the same 42 raw feature columns in any order —
    # the pipeline handles scaling and encoding automatically.
    print("\n" + "=" * 62)
    print("  STEP 5 — SAVE FITTED PIPELINE")
    print("=" * 62)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\nFitted pipeline saved : {MODEL_PATH}")
    print(f"File size             : {size_kb:.1f} KB")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 62)
    print("  DONE")
    print("=" * 62)
    print(
        "\n[OK] Training complete."
        "\n     Original CSV files were not modified."
        "\n\n     To reload the saved model later:"
        "\n       import joblib"
        f"\n       pipeline = joblib.load(r'{MODEL_PATH}')"
        "\n       y_pred   = pipeline.predict(X_new)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
