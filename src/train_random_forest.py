"""
train_random_forest.py
----------------------
Trains a Random Forest classifier on the UNSW-NB15 dataset using
the reusable preprocessing pipeline from data_preprocessing.py.

Fits only on training data.  Evaluates on the held-out test set.
Saves the fitted pipeline to models/random_forest_intrusion_detector.joblib.

The existing Logistic Regression model is NOT modified.
The original CSV files are NEVER modified.

Usage:
    python src/train_random_forest.py
"""

import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where Python is invoked from.
# sys.path is the list of directories Python searches for modules.
# Inserting src/ at position 0 ensures data_preprocessing.py is found
# before any other file with the same name elsewhere on the system.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import load_and_prepare  # noqa: E402

# ---------------------------------------------------------------------------
# Paths — derived from this file's location so they work on any machine.
# ROOT is the project root (one level above src/).
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "random_forest_intrusion_detector.joblib"


# ---------------------------------------------------------------------------
# Evaluation printer
# ---------------------------------------------------------------------------

def print_metrics(y_true, y_pred, y_prob) -> None:
    """
    Print all requested metrics with plain-English explanations.

    Parameters
    ----------
    y_true : array-like
        The correct labels from the test set (0 = normal, 1 = attack).
    y_pred : array-like
        The labels the model predicted for the test set.
    y_prob : array-like
        The model's estimated probability of class 1 (attack) per row.
        Used only for ROC-AUC — not needed for the other metrics.
    """

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    roc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])

    sep = "=" * 62

    print(f"\n{sep}")
    print("  INDIVIDUAL METRICS")
    print(sep)

    # -- Accuracy ------------------------------------------------------------
    # (correct predictions) / (all predictions).
    # Simple but misleading with imbalanced classes — a model that always
    # predicts "attack" would score ~55% without learning anything useful.
    print(f"\n  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(  "              Of ALL predictions, how many were correct?")

    # -- Precision (for attack class) ----------------------------------------
    # Of every connection flagged as an attack, what fraction really was one?
    # Low precision → many false alarms → analysts waste investigation time.
    print(f"\n  Precision : {prec:.4f}  ({prec * 100:.2f}%)")
    print(  "              Of predicted attacks, what fraction were real attacks?")

    # -- Recall (for attack class) -------------------------------------------
    # Of every real attack in the test set, what fraction did the model catch?
    # Low recall → real attacks slip through → the most dangerous failure mode.
    print(f"\n  Recall    : {rec:.4f}  ({rec * 100:.2f}%)")
    print(  "              Of ALL real attacks, what fraction did we catch?")

    # -- F1-score ------------------------------------------------------------
    # Harmonic mean of precision and recall:
    #   F1 = 2 * (precision * recall) / (precision + recall)
    # A single balanced score — penalises extreme sacrifice of either metric.
    print(f"\n  F1-score  : {f1:.4f}  ({f1 * 100:.2f}%)")
    print(  "              Balanced score combining precision and recall.")

    # -- ROC-AUC -------------------------------------------------------------
    # Probability that the model ranks a random attack ABOVE a random normal
    # connection.  Threshold-independent — measures general separability.
    #   0.5 = random guess   1.0 = perfect separation
    print(f"\n  ROC-AUC   : {roc:.4f}  ({roc * 100:.2f}%)")
    print(  "              Probability model ranks a random attack above a random")
    print(  "              normal connection.  0.5 = random, 1.0 = perfect.")

    # -- Confusion Matrix ----------------------------------------------------
    # A 2x2 table showing the four possible prediction outcomes:
    #
    #               | Predicted Normal  | Predicted Attack
    # --------------|-------------------|------------------
    # Actual Normal |  TN  (correct)    |  FP  (false alarm)
    # Actual Attack |  FN  (missed!)    |  TP  (correct)
    #
    #  TN = True  Negative : normal connection correctly identified as normal
    #  FP = False Positive : normal connection wrongly flagged as an attack
    #  FN = False Negative : real attack the model MISSED  <- minimise this
    #  TP = True  Positive : attack correctly identified as an attack
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"  {'':<28} {'Predicted Normal':>16}  {'Predicted Attack':>16}")
    print(f"  {'-' * 64}")
    print(f"  {'Actual Normal  (label=0)':<28} {tn:>16,}  {fp:>16,}")
    print(f"  {'Actual Attack  (label=1)':<28} {fn:>16,}  {tp:>16,}")
    print(f"  {'-' * 64}")
    print(f"  {'':28}  FP = false alarms      FN = missed attacks")

    # -- Classification Report -----------------------------------------------
    # Per-class breakdown of precision, recall, and F1 for BOTH classes.
    # "support" = how many rows of that class exist in the test set.
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
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Full training and evaluation workflow:

      Step 1 -- Load data via load_and_prepare() from data_preprocessing.py.
      Step 2 -- Add RandomForestClassifier as the final pipeline step.
      Step 3 -- Fit the complete pipeline on training data ONLY.
      Step 4 -- Evaluate on the held-out test set.
      Step 5 -- Save the fitted pipeline to disk with joblib.
    """

    # =========================================================================
    # STEP 1 — LOAD DATA
    # =========================================================================
    # load_and_prepare() handles all CSV loading, validation, X/y splitting,
    # and returns a fresh UNFITTED preprocessing pipeline.
    # We discard num_cols (captured as _) — it was only needed inside
    # data_preprocessing.py when wiring the ColumnTransformer.
    print("\n" + "=" * 62)
    print("  STEP 1 — LOAD DATA")
    print("=" * 62)

    X_train, X_test, y_train, y_test, pipeline, _ = load_and_prepare()

    # =========================================================================
    # STEP 2 — ADD RANDOM FOREST TO THE PIPELINE
    # =========================================================================
    # load_and_prepare() returns a pipeline that contains only the preprocessor.
    # We append the classifier as the final step so the whole chain (preprocess
    # + predict) is one single object.
    #
    # After appending the full pipeline looks like:
    #
    #   Pipeline
    #   +-- "preprocessor"  ColumnTransformer
    #   |   +-- "cat"  OneHotEncoder   (proto, service, state)
    #   |   +-- "num"  StandardScaler  (39 numerical columns)
    #   +-- "model"  RandomForestClassifier
    #
    # --- Hyperparameter choices explained ------------------------------------
    #
    #   n_estimators=150
    #       A Random Forest builds many independent Decision Trees and then
    #       takes a majority vote.  150 trees gives a good balance between
    #       accuracy and training time.  More trees → marginally better but
    #       slower; fewer trees → faster but noisier predictions.
    #
    #   max_depth=None
    #       Each tree is allowed to grow until every leaf is pure (contains
    #       only one class) or until min_samples_leaf stops it.  Setting a
    #       finite depth (e.g. 20) would make training faster but at the cost
    #       of accuracy.  None is the standard starting point.
    #
    #   min_samples_leaf=2
    #       A leaf node must contain at least 2 training samples.  This
    #       prevents the tree from memorising single noisy rows (overfitting).
    #       The default is 1, which often overfits on large datasets.
    #
    #   class_weight="balanced"
    #       The training set contains ~68% attacks and ~32% normal connections.
    #       "balanced" increases the penalty for misclassifying the minority
    #       class (normal), so the model is forced to learn both classes rather
    #       than always predicting "attack" for a cheap high accuracy.
    #       Weights are computed as:
    #           weight(class) = total_rows / (n_classes * count(class))
    #
    #   random_state=42
    #       Seeds all internal random operations (bootstrap sampling, feature
    #       selection at each split) so results are reproducible across runs.
    #
    #   n_jobs=-1
    #       Tells scikit-learn to use ALL available CPU cores when building
    #       trees in parallel.  With 150 trees this makes a significant
    #       difference to training time.
    #       Note: n_jobs is valid for RandomForestClassifier (it was only
    #       deprecated for LogisticRegression in scikit-learn 1.8+).
    # -------------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  STEP 2 — BUILD COMPLETE PIPELINE")
    print("=" * 62)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # pipeline.steps is a list of ("name", transformer/estimator) tuples.
    # Appending adds "model" as the new last step, then we rebuild the
    # Pipeline object so scikit-learn re-validates the full chain.
    pipeline.steps.append(("model", clf))
    pipeline = Pipeline(steps=pipeline.steps)

    print("\nComplete pipeline (unfitted):")
    print(pipeline)

    # =========================================================================
    # STEP 3 — FIT ON TRAINING DATA ONLY
    # =========================================================================
    # pipeline.fit(X_train, y_train) runs the two steps in order:
    #   a) preprocessor.fit_transform(X_train)
    #      -> learns mean/std from numerical columns in X_train
    #      -> learns category lists from proto/service/state in X_train
    #      -> converts X_train into a fully numeric matrix
    #   b) model.fit(transformed_X_train, y_train)
    #      -> builds 150 Decision Trees using bootstrap samples of X_train
    #         (each tree sees a different random 63% of rows and a random
    #          subset of features at each split)
    #
    # X_test and y_test are NOT seen during this step in any way.
    print("\n" + "=" * 62)
    print("  STEP 3 — FIT ON TRAINING DATA")
    print("=" * 62)
    print(f"\nTraining on {len(X_train):,} rows with {clf.n_estimators} trees ...")
    print("(Random Forest on 175k rows typically takes 1-3 minutes.")
    print(" Please wait ...)")

    t_start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t_start

    print(f"\nTraining complete in {elapsed:.1f} seconds.")

    # =========================================================================
    # STEP 4 — EVALUATE ON THE HELD-OUT TEST SET
    # =========================================================================
    # pipeline.predict(X_test):
    #   -> runs preprocessor.transform(X_test) using the ALREADY-FITTED scaler
    #      and encoder — NO new statistics are learned from the test set
    #   -> passes the result to the fitted RandomForest for predictions
    #   -> returns a 1-D array of 0s (normal) and 1s (attack)
    #
    # pipeline.predict_proba(X_test)[:, 1]:
    #   -> returns the fraction of the 150 trees that voted "attack" per row
    #   -> column 0 = P(normal), column 1 = P(attack)
    #   -> we take column 1 for ROC-AUC
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
    # joblib.dump() serialises the entire fitted pipeline (preprocessor +
    # Random Forest) to a single binary file.
    #
    # This is saved to a DIFFERENT path from the Logistic Regression model:
    #   LR model : models/intrusion_detector.joblib          (unchanged)
    #   RF model : models/random_forest_intrusion_detector.joblib  (new)
    #
    # To reload and use this model later without re-training:
    #   import joblib
    #   pipeline = joblib.load("models/random_forest_intrusion_detector.joblib")
    #   y_pred   = pipeline.predict(X_new)
    #
    # X_new only needs the same 42 raw feature columns — the pipeline handles
    # all scaling and encoding internally.
    print("\n" + "=" * 62)
    print("  STEP 5 — SAVE FITTED PIPELINE")
    print("=" * 62)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\nFitted pipeline saved : {MODEL_PATH}")
    print(f"File size             : {size_mb:.1f} MB")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 62)
    print("  DONE")
    print("=" * 62)
    print(
        "\n[OK] Training complete."
        "\n     Original CSV files were not modified."
        "\n     Logistic Regression model was not modified."
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
