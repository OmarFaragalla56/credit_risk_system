"""
tests/test_transformer.py
--------------------------
Pytest test suite for the CreditRiskFeatureEngineer custom transformer.

Why these tests matter:
  - They prove the transformer is production-safe before it ever touches real data.
  - They serve as regression tests — if someone refactors the class, any breaking
    change will immediately surface here instead of silently in production.
  - They demonstrate to interviewers that you write testable, reliable ML code.

Run with:
  pytest tests/test_transformer.py -v
  make test
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the app module importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from custom_transformers import CreditRiskFeatureEngineer

# ── Fixtures ───────────────────────────────────────────────────────────────────

KAGGLE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """
    Returns a realistic small training DataFrame with the original Kaggle column names.
    All values are within normal ranges.
    """
    data = {
        "RevolvingUtilizationOfUnsecuredLines": [0.10, 0.85, 0.50, 0.30, 0.95],
        "age":                                  [45,   23,   55,   38,   60],
        "NumberOfTime30-59DaysPastDueNotWorse": [0,    2,    0,    1,    0],
        "DebtRatio":                            [0.20, 0.75, 0.40, 1.20, 0.10],
        "MonthlyIncome":                        [5000.0, 2800.0, 7000.0, 4500.0, 25000.0],
        "NumberOfOpenCreditLinesAndLoans":      [5,    2,    10,   8,    15],
        "NumberOfTimes90DaysLate":              [0,    1,    0,    0,    0],
        "NumberRealEstateLoansOrLines":         [1,    0,    2,    1,    2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0,    1,    0,    0,    0],
        "NumberOfDependents":                   [1.0,  0.0,  2.0,  3.0,  1.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def fitted_transformer(sample_train_df) -> CreditRiskFeatureEngineer:
    """Returns a transformer fitted on the sample training data."""
    t = CreditRiskFeatureEngineer(cap_outliers=True)
    t.fit(sample_train_df)
    return t


# ── Test 1: TypeError on NumPy Input ──────────────────────────────────────────

def test_raises_on_numpy_array_input(fitted_transformer, sample_train_df):
    """
    FIX #1 VERIFICATION.
    Ensures the transformer hard-fails if a NumPy array is passed instead of a DataFrame.
    Without this, string-based column checks silently evaluate to False on integer column names.
    """
    numpy_input = sample_train_df.to_numpy()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        fitted_transformer.transform(numpy_input)


def test_raises_on_numpy_during_fit(sample_train_df):
    """Ensures fit() also rejects NumPy arrays."""
    t = CreditRiskFeatureEngineer()
    numpy_input = sample_train_df.to_numpy()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        t.fit(numpy_input)


# ── Test 2: Outlier Capping Respects 99th Percentile ─────────────────────────

def test_outlier_capping_respects_99th_percentile(sample_train_df):
    """
    FIX #2 VERIFICATION.
    Ensures that after transform(), no value in a numeric column exceeds the 99th
    percentile that was computed during fit().
    """
    t = CreditRiskFeatureEngineer(cap_outliers=True)
    t.fit(sample_train_df)

    # Inject an extreme outlier that should be clipped
    extreme_df = sample_train_df.copy()
    extreme_df.loc[0, "MonthlyIncome"] = 999_999_999.0  # Clearly above 99th pct

    transformed = t.transform(extreme_df)

    income_col = "MonthlyIncome"
    upper_bound = t.upper_bounds[income_col]
    assert transformed[income_col].max() <= upper_bound, (
        f"Expected MonthlyIncome max <= {upper_bound}, got {transformed[income_col].max()}"
    )


# ── Test 3: income_per_dependent Never Produces NaN ──────────────────────────

def test_income_per_dependent_no_nan_on_missing_income(sample_train_df, fitted_transformer):
    """
    FIX #3 VERIFICATION.
    Ensures that NaN MonthlyIncome does not propagate into the engineered
    'income_per_dependent' feature.
    """
    df_with_null = sample_train_df.copy()
    df_with_null.loc[0, "MonthlyIncome"] = np.nan

    transformed = fitted_transformer.transform(df_with_null)

    assert "income_per_dependent" in transformed.columns, "income_per_dependent feature is missing."
    null_count = transformed["income_per_dependent"].isnull().sum()
    assert null_count == 0, (
        f"income_per_dependent has {null_count} NaN(s) when MonthlyIncome contains nulls. "
        "Missing .fillna(0) on MonthlyIncome."
    )


# ── Test 4: Output Column Count Is Deterministic ─────────────────────────────

def test_output_shape_is_deterministic(sample_train_df, fitted_transformer):
    """
    Ensures the transformer always produces the same number of output columns,
    regardless of input row count. Non-determinism here would break the downstream
    ColumnTransformer which expects a fixed-size feature matrix.
    """
    single_row = sample_train_df.iloc[[0]]
    full_df = sample_train_df

    out_single = fitted_transformer.transform(single_row)
    out_full   = fitted_transformer.transform(full_df)

    assert out_single.shape[1] == out_full.shape[1], (
        f"Column count mismatch: single row has {out_single.shape[1]} cols, "
        f"full df has {out_full.shape[1]} cols."
    )
