"""
src/process.py
--------------
Data Processing Script — Stage 2 of the MLOps Pipeline.

What this script does:
  1. LOADS the raw batch CSV produced by ingest.py.
  2. SPLITS data into train/test sets (stratified on the target column).
     The split ALWAYS happens first — before any transformation — to guarantee
     zero data leakage between training and evaluation.
  3. FITS the CreditRiskFeatureEngineer + IterativeImputer + StandardScaler
     ONLY on training data, then transforms both train and test sets.
  4. SAVES the updated master_pipeline.pkl so the FastAPI backend always has
     the latest fitted preprocessing state.
  5. EXPORTS processed CSVs for train.py to consume.

Usage:
  python -m src.process
  make process
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (activates IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Path Resolution ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from custom_transformers import CreditRiskFeatureEngineer  # noqa: E402

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config Loading ─────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW_BATCH_PATH = PROJECT_ROOT / cfg["paths"]["raw_batch"]
TRAIN_OUT_PATH = PROJECT_ROOT / cfg["paths"]["processed_train"]
TEST_OUT_PATH  = PROJECT_ROOT / cfg["paths"]["processed_test"]
PIPELINE_PATH  = PROJECT_ROOT / cfg["paths"]["master_pipeline"]


def load_raw_batch() -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the validated raw batch and separates features from target.

    Returns
    -------
    X : pd.DataFrame
        Raw feature columns (no target, no unnamed index).
    y : pd.Series
        Binary target column (SeriousDlqin2yrs).
    """
    logger.info("Loading raw batch from: %s", RAW_BATCH_PATH)
    df = pd.read_csv(RAW_BATCH_PATH)

    # Rename target for consistency
    if "SeriousDlqin2yrs" in df.columns:
        df.rename(columns={"SeriousDlqin2yrs": "target"}, inplace=True)

    X = df.drop(columns=["target", "Unnamed: 0", "processed"], errors="ignore")
    y = df["target"]
    logger.info("Loaded %d rows, %d features.", len(X), X.shape[1])
    return X, y


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Assembles and fits the master preprocessing pipeline on training data only.

    Pipeline steps:
      1. CreditRiskFeatureEngineer: caps outliers (99th pct) + engineers domain features.
         Fitted on training data — upper_bounds learned here are NEVER exposed to test data.
      2. ColumnTransformer with IterativeImputer + StandardScaler applied to all numeric cols.
         IterativeImputer models each column as a function of all others (Ridge regression),
         which is far superior to naive median imputation for correlated financial data.

    Returns the fitted Pipeline object.
    """
    feature_engineer = CreditRiskFeatureEngineer(cap_outliers=True)

    # Dry-run to discover the output column set after feature engineering
    dummy_out = feature_engineer.fit_transform(X_train.head(10))
    engineered_cols = dummy_out.columns.tolist()

    # Re-instantiate so fit() runs fresh on the full training data
    feature_engineer = CreditRiskFeatureEngineer(cap_outliers=True)

    num_pipeline = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
        ("scaler", StandardScaler()),
    ])

    master_pipeline = Pipeline([
        ("feature_engineering", feature_engineer),
        ("scaling", ColumnTransformer([
            ("num", num_pipeline, engineered_cols)
        ], remainder="passthrough", verbose_feature_names_out=False)),
    ])

    logger.info("Fitting master pipeline on %d training samples...", len(X_train))
    master_pipeline.fit(X_train)
    logger.info("Pipeline fitted successfully.")
    return master_pipeline


def save_outputs(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persists the fitted pipeline and processed data splits to disk."""
    # Save pipeline
    os.makedirs(PIPELINE_PATH.parent, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info("Saved master_pipeline.pkl to: %s", PIPELINE_PATH)

    # Transform both splits
    X_train_proc = pipeline.transform(X_train)
    X_test_proc  = pipeline.transform(X_test)

    try:
        feature_names = pipeline.named_steps["scaling"].get_feature_names_out().tolist()
    except Exception:
        feature_names = [str(i) for i in range(X_train_proc.shape[1])]

    # Save processed CSVs
    os.makedirs(TRAIN_OUT_PATH.parent, exist_ok=True)
    train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    train_df["target"] = y_train.values
    train_df.to_csv(TRAIN_OUT_PATH, index=False)
    logger.info("Saved processed training data (%d rows) to: %s", len(train_df), TRAIN_OUT_PATH)

    test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
    test_df["target"] = y_test.values
    test_df.to_csv(TEST_OUT_PATH, index=False)
    logger.info("Saved processed test data (%d rows) to: %s", len(test_df), TEST_OUT_PATH)


def main():
    logger.info("=" * 60)
    logger.info("STAGE 2: Data Processing — Starting")
    logger.info("=" * 60)

    # Load raw data
    X, y = load_raw_batch()

    # SPLIT FIRST — this is non-negotiable. Prevents leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(
        "Train/test split: %d train rows, %d test rows (stratified).",
        len(X_train), len(X_test)
    )

    # Fit pipeline on training data only, then transform both
    pipeline = build_pipeline(X_train)
    save_outputs(pipeline, X_train, X_test, y_train, y_test)

    logger.info("=" * 60)
    logger.info("STAGE 2: Data Processing — Complete ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
