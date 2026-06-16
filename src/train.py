"""
src/train.py
-------------
Model Training Script — Stage 3 of the MLOps Pipeline.
Trains a blended XGBoost + LightGBM ensemble for optimal credit risk prediction.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path Resolution ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from custom_transformers import CreditRiskEnsemble

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config Loading ─────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

TRAIN_PATH       = PROJECT_ROOT / cfg["paths"]["processed_train"]
TEST_PATH        = PROJECT_ROOT / cfg["paths"]["processed_test"]
CHAMPION_PATH    = PROJECT_ROOT / cfg["paths"]["champion_model"]
OPTUNA_TRIALS    = cfg["model"]["optuna_trials"]
MLFLOW_URI       = cfg["mlflow"]["tracking_uri"]
EXPERIMENT_NAME  = cfg["mlflow"]["experiment_name"]
REGISTERED_NAME  = cfg["mlflow"]["registered_model_name"]


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads the processed train/test CSVs produced by process.py."""
    logger.info("Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test  = test_df.drop(columns=["target"])
    y_test  = test_df["target"]

    logger.info("Train: %d rows | Test: %d rows | Features: %d", len(X_train), len(X_test), X_train.shape[1])
    return X_train, X_test, y_train, y_test


def run_xgb_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mlflow_run_id: str,
) -> tuple[dict, float]:
    """Runs Optuna hyperparameter search for XGBoost using 5-Fold Stratified CV."""
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    scale_pos_weight = neg_class_count / pos_class_count
    logger.info("XGBoost class imbalance ratio: %.2f", scale_pos_weight)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      150,
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2),
            "scale_pos_weight":  scale_pos_weight,
            "eval_metric":       "auc",
            "random_state":      42,
            "n_jobs":            -1,
            "early_stopping_rounds": 10,
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
            
        cv_score = np.mean(scores)
        with mlflow.start_run(run_id=mlflow_run_id, nested=True):
            mlflow.log_params({f"xgb_trial_{trial.number}_{k}": v for k, v in params.items()})
            mlflow.log_metric(f"xgb_trial_{trial.number}_cv_roc_auc", cv_score)
        return cv_score

    study = optuna.create_study(direction="maximize")
    logger.info("Starting XGBoost Optuna study...")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("Best XGBoost CV ROC-AUC: %.4f", study.best_value)
    return study.best_params, study.best_value


def run_lgb_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mlflow_run_id: str,
) -> tuple[dict, float]:
    """Runs Optuna hyperparameter search for LightGBM using 5-Fold Stratified CV."""
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    scale_pos_weight = neg_class_count / pos_class_count
    logger.info("LightGBM class imbalance ratio: %.2f", scale_pos_weight)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      150,
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "objective":         "binary",
            "metric":            "auc",
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(stopping_rounds=10, verbose=False)]
            )
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
            
        cv_score = np.mean(scores)
        with mlflow.start_run(run_id=mlflow_run_id, nested=True):
            mlflow.log_params({f"lgb_trial_{trial.number}_{k}": v for k, v in params.items()})
            mlflow.log_metric(f"lgb_trial_{trial.number}_cv_roc_auc", cv_score)
        return cv_score

    study = optuna.create_study(direction="maximize")
    logger.info("Starting LightGBM Optuna study...")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("Best LightGBM CV ROC-AUC: %.4f", study.best_value)
    return study.best_params, study.best_value


def train_xgb_champion(
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    neg_class_count = (y_train == 0).sum()
    scale_pos_weight = neg_class_count / (y_train == 1).sum()

    final_params = {
        **best_params,
        "n_estimators": 300,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 10,
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    logger.info("Fitting final XGBoost model...")
    model = XGBClassifier(**final_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_champion(
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LGBMClassifier:
    neg_class_count = (y_train == 0).sum()
    scale_pos_weight = neg_class_count / (y_train == 1).sum()

    final_params = {
        **best_params,
        "n_estimators": 300,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    logger.info("Fitting final LightGBM model...")
    model = LGBMClassifier(**final_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=10, verbose=False)]
    )
    return model


def main():
    logger.info("=" * 60)
    logger.info("STAGE 3: Model Training (Ensemble Blending) — Starting")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_splits()

    if "://" in MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)
    else:
        mlflow.set_tracking_uri((PROJECT_ROOT / MLFLOW_URI).as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="optuna_ensemble_challenger") as run:
        run_id = run.info.run_id
        logger.info("MLflow Run ID: %s", run_id)

        # 1. Optimize XGBoost
        xgb_best_params, xgb_best_score = run_xgb_optuna_study(X_train, y_train, run_id)
        xgb_model = train_xgb_champion(xgb_best_params, X_train, y_train)

        # 2. Optimize LightGBM
        lgb_best_params, lgb_best_score = run_lgb_optuna_study(X_train, y_train, run_id)
        lgb_model = train_lgb_champion(lgb_best_params, X_train, y_train)

        # 3. Create Ensemble (50% XGBoost, 50% LightGBM)
        ensemble = CreditRiskEnsemble(xgb_model, lgb_model, xgb_weight=0.5)

        # Evaluate Individual models vs Ensemble
        xgb_test_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
        lgb_test_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
        ensemble_test_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])

        logger.info("XGBoost alone ROC-AUC: %.4f", xgb_test_auc)
        logger.info("LightGBM alone ROC-AUC: %.4f", lgb_test_auc)
        logger.info("🏆 Ensemble ROC-AUC: %.4f", ensemble_test_auc)

        # Log metrics to MLflow
        mlflow.log_metric("xgb_test_roc_auc", xgb_test_auc)
        mlflow.log_metric("lgb_test_roc_auc", lgb_test_auc)
        mlflow.log_metric("roc_auc", ensemble_test_auc)

        # Register individual models
        mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_model")
        mlflow.lightgbm.log_model(lgb_model, artifact_path="lgb_model")

        # Save the blended ensemble model to disk
        challenger_path = PROJECT_ROOT / "models" / "challenger_xgboost.pkl"
        joblib.dump(ensemble, challenger_path)
        logger.info("Saved blended challenger ensemble model to: %s", challenger_path)

        mlflow.set_tag("challenger_roc_auc", f"{ensemble_test_auc:.4f}")

    logger.info("=" * 60)
    logger.info("STAGE 3: Model Training — Complete ✅")
    logger.info("Ensemble ROC-AUC: %.4f", ensemble_test_auc)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
