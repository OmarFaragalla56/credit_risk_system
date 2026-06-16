"""
src/evaluate.py
---------------
Model Evaluation & Promotion Script — Stage 4 of the MLOps Pipeline.

What this script does:
  1. LOADS the challenger model produced by train.py and the current production champion.
  2. COMPARES their ROC-AUC scores on the same held-out test set.
  3. PROMOTES the challenger to production (overwrites champion_xgboost.pkl) IF it wins.
  4. LOGS the promotion decision to MLflow as a tag for audit trail purposes.
     In a real production system, this would also trigger a Docker image rebuild
     and a Kubernetes rolling deployment.

Usage:
  python -m src.evaluate
  make promote
"""

import logging
import os
import sys
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

# ── Path Resolution ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from custom_transformers import CreditRiskFeatureEngineer  # noqa: F401 (required for joblib)

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config Loading ─────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

TEST_PATH         = PROJECT_ROOT / cfg["paths"]["processed_test"]
CHAMPION_PATH     = PROJECT_ROOT / cfg["paths"]["champion_model"]
CHALLENGER_PATH   = PROJECT_ROOT / "models" / "challenger_xgboost.pkl"
MLFLOW_URI        = cfg["mlflow"]["tracking_uri"]
EXPERIMENT_NAME   = cfg["mlflow"]["experiment_name"]


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the processed held-out test set for head-to-head model comparison."""
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    logger.info("Loaded test set: %d rows, %d features.", len(X_test), X_test.shape[1])
    return X_test, y_test


def score_model(model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> float:
    """
    Evaluates a model's ROC-AUC on the test set.
    Dynamically subsets features if the model expects a different schema (e.g. older champion).
    """
    model_features = None
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, "feature_names"):
        model_features = list(model.feature_names)
    elif hasattr(model, "xgb_model") and hasattr(model.xgb_model, "feature_names_in_"):
        model_features = list(model.xgb_model.feature_names_in_)

    if model_features is not None:
        # Check if all model features exist in X_test
        missing = [f for f in model_features if f not in X_test.columns]
        if not missing:
            X_test_eval = X_test[model_features]
        else:
            logger.warning(
                "%s expects features %s that are missing from X_test. Using raw X_test.",
                label, missing
            )
            X_test_eval = X_test
    else:
        X_test_eval = X_test

    prob = model.predict_proba(X_test_eval)[:, 1]
    score = roc_auc_score(y_test, prob)
    logger.info("%s ROC-AUC: %.4f", label, score)
    return score


def main():
    logger.info("=" * 60)
    logger.info("STAGE 4: Model Evaluation & Promotion — Starting")
    logger.info("=" * 60)

    # Load test data
    X_test, y_test = load_test_data()

    # Load both models
    if not CHALLENGER_PATH.exists():
        logger.error("No challenger model found at %s. Run train.py first.", CHALLENGER_PATH)
        sys.exit(1)

    challenger = joblib.load(CHALLENGER_PATH)
    logger.info("Loaded challenger model from: %s", CHALLENGER_PATH)

    if not CHAMPION_PATH.exists():
        logger.warning("No champion model found. Promoting challenger by default.")
        joblib.dump(challenger, CHAMPION_PATH)
        logger.info("Challenger promoted to champion: %s", CHAMPION_PATH)
        return

    champion = joblib.load(CHAMPION_PATH)
    logger.info("Loaded champion model from: %s", CHAMPION_PATH)

    # Score both models on the same test set
    champion_score   = score_model(champion, X_test, y_test, "Champion")
    challenger_score = score_model(challenger, X_test, y_test, "Challenger")

    # Log comparison to MLflow for audit trail
    if "://" in MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)
    else:
        mlflow.set_tracking_uri((PROJECT_ROOT / MLFLOW_URI).as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model_promotion_evaluation"):
        mlflow.log_metric("champion_roc_auc", champion_score)
        mlflow.log_metric("challenger_roc_auc", challenger_score)
        mlflow.log_metric("roc_auc_delta", challenger_score - champion_score)

        # ── The Promotion Decision ─────────────────────────────────────────────
        if challenger_score > champion_score:
            improvement = challenger_score - champion_score
            logger.info(
                "🏆 Challenger WINS (+%.4f ROC-AUC). Promoting to production...",
                improvement,
            )
            joblib.dump(challenger, CHAMPION_PATH)
            mlflow.set_tag("promotion_decision", "CHALLENGER_PROMOTED")
            mlflow.set_tag("improvement", f"+{improvement:.4f}")
            logger.info("New champion saved to: %s", CHAMPION_PATH)
            logger.info("🚀 FastAPI backend will use the new model on next restart.")
        else:
            gap = champion_score - challenger_score
            logger.info(
                "🛡️  Champion HOLDS (challenger is %.4f worse). No promotion.",
                gap,
            )
            mlflow.set_tag("promotion_decision", "CHAMPION_RETAINED")
            mlflow.set_tag("performance_gap", f"-{gap:.4f}")

    logger.info("=" * 60)
    logger.info("STAGE 4: Model Evaluation & Promotion — Complete ✅")
    logger.info("Champion: %.4f | Challenger: %.4f", champion_score, challenger_score)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
