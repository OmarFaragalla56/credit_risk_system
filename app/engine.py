"""
engine.py
---------
Inference engine for the Credit Risk Intelligence API.

Loads the two-stage production model:
  1. master_pipeline.pkl  — The full preprocessing pipeline (CreditRiskFeatureEngineer
                            + IterativeImputer + StandardScaler), fitted on training data.
  2. champion_xgboost.pkl — The Optuna-optimized XGBoost model (best ROC-AUC from study).

Exposes three methods for the FastAPI layer:
  - get_prediction()    : Returns raw default probability.
  - get_reasons()       : Returns top-5 SHAP feature contributions.
  - get_roadmap_to_yes(): Returns a counterfactual recommendation.
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

# --- Make the custom transformer importable when loading the pipeline pickle ---
# joblib/pickle needs CreditRiskFeatureEngineer to be resolvable at load time.
sys.path.insert(0, os.path.dirname(__file__))
from custom_transformers import CreditRiskFeatureEngineer, CreditRiskEnsemble  # noqa: F401 (needed for pickle)

logger = logging.getLogger(__name__)

# Mapping from the Kaggle raw column names to the user-friendly Streamlit API keys.
# The Streamlit frontend sends short keys (e.g. "util_ratio"); the master_pipeline
# was trained on original Kaggle column names (e.g. "RevolvingUtilizationOfUnsecuredLines").
COLUMN_MAP = {
    "util_ratio":        "RevolvingUtilizationOfUnsecuredLines",
    "age":               "age",
    "late_30_59":        "NumberOfTime30-59DaysPastDueNotWorse",
    "late_60_89":        "NumberOfTime60-89DaysPastDueNotWorse",
    "late_90_plus":      "NumberOfTimes90DaysLate",
    "debt_ratio":        "DebtRatio",
    "monthly_income":    "MonthlyIncome",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "dependents":        "NumberOfDependents",
}


class InferenceEngine:
    """
    Two-stage inference engine: preprocessing pipeline → XGBoost champion model.

    Parameters
    ----------
    pipeline_path : str
        Path to master_pipeline.pkl (the fitted sklearn Pipeline).
    model_path : str
        Path to champion_xgboost.pkl (the fitted XGBoost Booster/classifier).
    """

    def __init__(self, pipeline_path: str, model_path: str):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found: {pipeline_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Champion model not found: {model_path}")

        logger.info("Loading preprocessing pipeline from: %s", pipeline_path)
        self.pipeline = joblib.load(pipeline_path)

        logger.info("Loading champion model from: %s", model_path)
        self.model = joblib.load(model_path)

        # Build SHAP explainer(s)
        if hasattr(self.model, "xgb_model") and hasattr(self.model, "lgb_model"):
            logger.info("Ensemble model detected. Building XGBoost and LightGBM tree explainers...")
            self.xgb_explainer = shap.TreeExplainer(self.model.xgb_model)
            self.lgb_explainer = shap.TreeExplainer(self.model.lgb_model)
            self.is_ensemble = True
        else:
            logger.info("Single tree model detected. Building tree explainer...")
            self.explainer = shap.TreeExplainer(self.model)
            self.is_ensemble = False
        logger.info("InferenceEngine initialized successfully.")

    def _prepare_input(self, raw_dict: dict) -> pd.DataFrame:
        """
        Converts the Streamlit API payload (short keys) into a DataFrame
        with the original Kaggle column names that the pipeline expects.
        """
        renamed = {COLUMN_MAP[k]: v for k, v in raw_dict.items() if k in COLUMN_MAP}
        df = pd.DataFrame([renamed])
        logger.debug("Prepared input DataFrame with columns: %s", df.columns.tolist())
        return df

    def get_prediction(self, raw_dict: dict) -> float:
        """
        Full inference: raw API payload → preprocessing → XGBoost → probability of default.

        Returns
        -------
        float
            Probability of serious delinquency (0.0 to 1.0).
        """
        input_df = self._prepare_input(raw_dict)
        processed = self.pipeline.transform(input_df)

        # pipeline.transform() returns a numpy array; wrap back to DataFrame for SHAP
        feature_names = self._get_feature_names()
        processed_df = pd.DataFrame(processed, columns=feature_names)

        prob = float(self.model.predict_proba(processed_df)[0, 1])
        logger.info("Prediction: probability of default = %.4f", prob)
        return prob

    def get_reasons(self, raw_dict: dict) -> list:
        """
        Computes top-5 SHAP feature contributions for the given borrower.

        Returns
        -------
        list of dict: [{"feature": str, "impact": float}, ...]
            Sorted by absolute impact descending.
        """
        input_df = self._prepare_input(raw_dict)
        processed = self.pipeline.transform(input_df)
        feature_names = self._get_feature_names()
        processed_df = pd.DataFrame(processed, columns=feature_names)

        if self.is_ensemble:
            xgb_shap = self.xgb_explainer(processed_df).values[0]
            lgb_shap = self.lgb_explainer(processed_df).values[0]
            contributions = self.model.xgb_weight * xgb_shap + (1 - self.model.xgb_weight) * lgb_shap
        else:
            shap_values = self.explainer(processed_df)
            contributions = shap_values.values[0]

        impact_list = [
            {
                "feature": name.replace("_", " ").title(),
                "impact": float(val),
                "abs_val": abs(float(val)),
            }
            for name, val in zip(feature_names, contributions)
        ]

        top_5 = sorted(impact_list, key=lambda x: x["abs_val"], reverse=True)[:5]
        return [{"feature": item["feature"], "impact": item["impact"]} for item in top_5]

    def get_roadmap_to_yes(self, raw_dict: dict, current_prob: float) -> str:
        """
        Dynamic counterfactual search: finds the minimum single-factor change
        that would push the borrower below the approval threshold.

        Parameters
        ----------
        raw_dict : dict
            Original raw API payload.
        current_prob : float
            Already-computed probability of default.
        """
        threshold = 0.30

        if current_prob <= threshold:
            return "Status: Approved. Maintain current credit utilization and payment history."

        # Strategy 1: Reduce utilization ratio
        for reduction in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            test_dict = raw_dict.copy()
            test_dict["util_ratio"] = raw_dict["util_ratio"] * (1 - reduction)
            new_prob = self.get_prediction(test_dict)
            if new_prob <= threshold:
                return (
                    f"Recommendation: Reducing credit utilization by {int(reduction * 100)}% "
                    "shifts the probability of default below the approval threshold."
                )

        # Strategy 2: Reduce debt ratio
        for reduction in [0.15, 0.30, 0.45]:
            test_dict = raw_dict.copy()
            test_dict["debt_ratio"] = raw_dict["debt_ratio"] * (1 - reduction)
            new_prob = self.get_prediction(test_dict)
            if new_prob <= threshold:
                return (
                    f"Recommendation: A {int(reduction * 100)}% reduction in your total "
                    "debt-to-income ratio is required to meet approval criteria."
                )

        return (
            "Recommendation: Single-factor reduction is insufficient. A comprehensive "
            "strategy addressing both payment history and total debt is required."
        )

    def _get_feature_names(self) -> list:
        """
        Extracts feature names from the fitted pipeline's final ColumnTransformer step.
        Falls back to integer indices if names are unavailable.
        """
        try:
            # The pipeline structure: feature_engineering → scaling (ColumnTransformer)
            ct = self.pipeline.named_steps["scaling"]
            return ct.get_feature_names_out().tolist()
        except Exception:
            logger.warning("Could not extract feature names from pipeline; using indices.")
            return [str(i) for i in range(self.pipeline.transform(
                pd.DataFrame([{v: 0 for v in COLUMN_MAP.values()}])
            ).shape[1])]