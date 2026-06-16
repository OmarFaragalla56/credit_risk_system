"""
custom_transformers.py
----------------------
Production-grade Scikit-Learn custom transformer for the Credit Risk pipeline.
This module must be importable anywhere the master_pipeline.pkl is loaded.

Hotfixes applied:
  1. Raises TypeError if input is not a Pandas DataFrame (prevents silent NumPy trap).
  2. Uses pd.api.types.is_numeric_dtype() for robust outlier capping.
  3. Chains .fillna(0) on MonthlyIncome in income_per_dependent to prevent NaN propagation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer for credit risk feature engineering.

    Responsibilities:
      - Caps outliers at the 99th percentile (fitted ONLY on training data to prevent leakage).
      - Engineers domain-specific features: log transforms, utilization ratios,
        delinquency aggregates, and absolute monthly debt.

    Parameters
    ----------
    cap_outliers : bool, default=True
        Whether to clip numeric columns to their 99th percentile from the training set.
    """

    def __init__(self, cap_outliers: bool = True):
        self.cap_outliers = cap_outliers
        self.upper_bounds: dict = {}

    def fit(self, X, y=None):
        """
        Learn the 99th-percentile upper bounds from the TRAINING data only.
        This is the key anti-leakage mechanism — test data never influences these bounds.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"CreditRiskFeatureEngineer.fit() expects a pandas DataFrame, "
                f"but received {type(X).__name__}. Ensure .set_output(transform='pandas') "
                "is set on all prior pipeline steps."
            )

        if self.cap_outliers:
            for col in X.columns:
                # FIX #2: use is_numeric_dtype to catch float32, int32, Int64, etc.
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.upper_bounds[col] = X[col].quantile(0.99)
                    logger.debug("Outlier cap for '%s': %.4f", col, self.upper_bounds[col])

        logger.info("CreditRiskFeatureEngineer fitted on %d columns.", len(self.upper_bounds))
        return self

    def transform(self, X):
        """
        Apply outlier capping and engineer new features.
        Input MUST be a Pandas DataFrame with the original Kaggle column names.
        """
        # FIX #1: Hard fail on NumPy arrays instead of silently producing wrong features.
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"CreditRiskFeatureEngineer.transform() expects a pandas DataFrame, "
                f"but received {type(X).__name__}. Ensure .set_output(transform='pandas') "
                "is set on all prior pipeline steps."
            )

        X_out = X.copy()

        # --- Outlier Capping ---
        if self.cap_outliers:
            for col, upper_bound in self.upper_bounds.items():
                if col in X_out.columns:
                    X_out[col] = np.clip(X_out[col], a_min=None, a_max=upper_bound)

        # --- Feature Engineering ---

        # Log transform on monthly income to normalize the heavy right skew
        if "MonthlyIncome" in X_out.columns:
            X_out["log_monthly_income"] = np.log1p(X_out["MonthlyIncome"].fillna(0))

        # Utilization normalized by income — captures "is this person overextended?"
        if "RevolvingUtilizationOfUnsecuredLines" in X_out.columns and "MonthlyIncome" in X_out.columns:
            X_out["util_per_income"] = (
                X_out["RevolvingUtilizationOfUnsecuredLines"]
                / (X_out["MonthlyIncome"].fillna(0) + 1)
            )

        # Delinquency aggregation — total overdue events across all severity buckets
        delinq_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
        if all(c in X_out.columns for c in delinq_cols):
            X_out["total_past_due_events"] = X_out[delinq_cols].sum(axis=1)
            X_out["has_been_late"] = (X_out["total_past_due_events"] > 0).astype(int)

        # Income per dependent — measures real household financial pressure
        if "MonthlyIncome" in X_out.columns and "NumberOfDependents" in X_out.columns:
            # FIX #3: fillna(0) on MonthlyIncome to prevent NaN propagation here
            X_out["income_per_dependent"] = (
                X_out["MonthlyIncome"].fillna(0)
                / (X_out["NumberOfDependents"].fillna(0) + 1)
            )

        # Absolute monthly debt in USD — more interpretable than a ratio alone
        if "DebtRatio" in X_out.columns and "MonthlyIncome" in X_out.columns:
            conditions = [
                X_out["MonthlyIncome"].isna(),
                X_out["MonthlyIncome"].notna(),
            ]
            choices = [
                X_out["DebtRatio"],
                X_out["DebtRatio"] * X_out["MonthlyIncome"],
            ]
            X_out["absolute_monthly_debt"] = np.select(conditions, choices, default=0)

        # --- Advanced Domain Features ---
        if "RevolvingUtilizationOfUnsecuredLines" in X_out.columns:
            X_out["is_util_over_1"] = (X_out["RevolvingUtilizationOfUnsecuredLines"] > 1.0).astype(int)
            X_out["is_util_zero"] = (X_out["RevolvingUtilizationOfUnsecuredLines"] == 0.0).astype(int)
            
        if "absolute_monthly_debt" in X_out.columns and "MonthlyIncome" in X_out.columns:
            X_out["clean_debt_ratio"] = X_out["absolute_monthly_debt"] / (X_out["MonthlyIncome"].fillna(0) + 1)
            
        if "total_past_due_events" in X_out.columns and "NumberOfOpenCreditLinesAndLoans" in X_out.columns:
            X_out["late_events_per_line"] = X_out["total_past_due_events"] / (X_out["NumberOfOpenCreditLinesAndLoans"] + 1)
            
        if "NumberRealEstateLoansOrLines" in X_out.columns and "NumberOfOpenCreditLinesAndLoans" in X_out.columns:
            X_out["secured_vs_unsecured_ratio"] = X_out["NumberRealEstateLoansOrLines"] / (X_out["NumberOfOpenCreditLinesAndLoans"] + 1)

        logger.info(
            "CreditRiskFeatureEngineer.transform() produced %d features from %d input columns.",
            X_out.shape[1],
            X.shape[1],
        )
        return X_out


class CreditRiskEnsemble:
    """
    Production-grade model ensemble combining XGBoost and LightGBM models.
    """
    def __init__(self, xgb_model, lgb_model, xgb_weight=0.5):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.xgb_weight = xgb_weight

    def predict_proba(self, X):
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        ensemble_prob = self.xgb_weight * xgb_prob + (1 - self.xgb_weight) * lgb_prob
        return np.column_stack([1 - ensemble_prob, ensemble_prob])

    def predict(self, X, threshold=0.30):
        # Default decision threshold (e.g. 0.30 from config risk_threshold)
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

