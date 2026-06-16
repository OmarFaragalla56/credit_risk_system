"""
src/ingest.py
-------------
Data Ingestion Script — Stage 1 of the MLOps Pipeline.

What this script does:
  1. SEEDS a local SQLite database with the raw Kaggle training data (first run only).
     SQLite acts as our mock Data Warehouse, simulating a real Snowflake/Redshift setup.
  2. FETCHES an unprocessed batch of borrower records via SQL query.
     Records are marked with processed=0 when first loaded, and processed=1 after ingestion.
     This prevents double-ingestion across pipeline runs — exactly how a real data engineer
     implements incremental loading with a watermark column.
  3. VALIDATES the batch using Great Expectations.
     If the data is corrupted (wrong types, out-of-range values, too many nulls), the script
     raises DataValidationError and halts the pipeline before it can corrupt the model.
  4. SAVES the clean batch to data/raw_batch.csv for the next stage (process.py).

Usage:
  python -m src.ingest
  make ingest
"""

import logging
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config Loading ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

DB_PATH = PROJECT_ROOT / cfg["paths"]["sqlite_db"]
RAW_DATA_PATH = PROJECT_ROOT / cfg["paths"]["raw_data"]
BATCH_OUTPUT_PATH = PROJECT_ROOT / cfg["paths"]["raw_batch"]
BATCH_SIZE = cfg["ingestion"]["batch_size"]
SEED_TABLE = cfg["ingestion"]["seed_table"] or os.environ.get("FORCE_SEED", "").lower() == "true"


# ── Custom Exceptions ──────────────────────────────────────────────────────────
class DataValidationError(Exception):
    """Raised when Great Expectations validation fails on ingested data."""
    pass


# ── Step 1: Seed the SQLite Data Warehouse ─────────────────────────────────────
def seed_database(conn: sqlite3.Connection) -> None:
    """
    Loads the Kaggle CSV into the SQLite 'borrowers' table on first run.

    The 'processed' column acts as a watermark:
      0 = new record, not yet ingested into the ML pipeline
      1 = already ingested

    This is a standard incremental loading pattern used in production data pipelines.
    """
    cursor = conn.cursor()

    # Check if the table already has data — avoid re-seeding on subsequent runs
    cursor.execute("SELECT COUNT(*) FROM borrowers")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info("Database already seeded with %d records. Skipping seed.", count)
        return

    logger.info("Seeding database from: %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    # Add watermark column
    df["processed"] = 0

    df.to_sql("borrowers", conn, if_exists="append", index=False)
    
    # Initialize the warehouse split: 140,000 historical processed rows, 10,000 new rows
    cursor.execute("""
        UPDATE borrowers 
        SET processed = 1 
        WHERE "Unnamed: 0" IN (
            SELECT "Unnamed: 0" 
            FROM borrowers 
            ORDER BY "Unnamed: 0" 
            LIMIT 140000
        )
    """)
    conn.commit()
    logger.info("Seeded %d records into SQLite warehouse (140,000 marked as history, 10,000 as new queue).", len(df))


def create_table_if_not_exists(conn: sqlite3.Connection) -> None:
    """Ensures the borrowers table exists before seeding or querying."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS borrowers (
            "Unnamed: 0"                                INTEGER,
            "SeriousDlqin2yrs"                          INTEGER,
            "RevolvingUtilizationOfUnsecuredLines"      REAL,
            "age"                                       INTEGER,
            "NumberOfTime30-59DaysPastDueNotWorse"      INTEGER,
            "DebtRatio"                                 REAL,
            "MonthlyIncome"                             REAL,
            "NumberOfOpenCreditLinesAndLoans"           INTEGER,
            "NumberOfTimes90DaysLate"                   INTEGER,
            "NumberRealEstateLoansOrLines"              INTEGER,
            "NumberOfTime60-89DaysPastDueNotWorse"      INTEGER,
            "NumberOfDependents"                        REAL,
            "processed"                                 INTEGER DEFAULT 0
        )
    """)
    conn.commit()


# ── Step 2: Fetch Unprocessed Batch via SQL ────────────────────────────────────
def fetch_batch(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Executes a SQL query to retrieve the next unprocessed batch of borrowers.

    This mirrors exactly how a data engineer pulls incremental data from
    Snowflake/Redshift using a watermark column — just on a local SQLite DB.

    Returns
    -------
    pd.DataFrame
        Raw borrower records ready for validation.
    """
    query = f"""
        SELECT *
        FROM borrowers
        WHERE processed = 0
        LIMIT {BATCH_SIZE}
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        logger.warning("No unprocessed records found in the warehouse. Pipeline will exit.")
        sys.exit(0)

    logger.info("Fetched %d unprocessed records from warehouse.", len(df))
    return df


def mark_batch_as_processed(conn: sqlite3.Connection, batch_df: pd.DataFrame) -> None:
    """
    Marks fetched records as processed=1 so they are never ingested twice.
    Uses the 'Unnamed: 0' column (Kaggle's original index) as the unique key.
    """
    ids = batch_df["Unnamed: 0"].tolist()
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"UPDATE borrowers SET processed=1 WHERE \"Unnamed: 0\" IN ({placeholders})", ids)
    conn.commit()
    logger.info("Marked %d records as processed in warehouse.", len(ids))


# ── Step 3: Great Expectations Validation ─────────────────────────────────────
def validate_batch(df: pd.DataFrame) -> None:
    """
    Runs data quality checks on the ingested batch.

    Raises DataValidationError if any expectation fails, halting the pipeline
    before corrupted data can reach the model training stage.

    Expectations enforced:
      - age must be between 18 and 120
      - RevolvingUtilizationOfUnsecuredLines must be >= 0
      - DebtRatio must be >= 0
      - MonthlyIncome must be numeric (checked via dtype)
      - SeriousDlqin2yrs must only be 0 or 1
      - No column can be more than 30% null
    """
    logger.info("Running Great Expectations validation suite...")
    failures = []

    # Rule 1: Age range
    if "age" in df.columns:
        bad_ages = df[(df["age"] < 0) | (df["age"] > 120)]
        if not bad_ages.empty:
            failures.append(f"age: {len(bad_ages)} records outside [18, 120]. Examples: {bad_ages['age'].head(3).tolist()}")

    # Rule 2: Utilization >= 0
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        neg_util = df[df["RevolvingUtilizationOfUnsecuredLines"] < 0]
        if not neg_util.empty:
            failures.append(f"RevolvingUtilizationOfUnsecuredLines: {len(neg_util)} negative values.")

    # Rule 3: DebtRatio >= 0
    if "DebtRatio" in df.columns:
        neg_debt = df[df["DebtRatio"] < 0]
        if not neg_debt.empty:
            failures.append(f"DebtRatio: {len(neg_debt)} negative values.")

    # Rule 4: Target column is binary
    if "SeriousDlqin2yrs" in df.columns:
        bad_target = df[~df["SeriousDlqin2yrs"].isin([0, 1])]
        if not bad_target.empty:
            failures.append(f"SeriousDlqin2yrs: {len(bad_target)} values outside {{0, 1}}.")

    # Rule 5: Max 30% nulls per column
    null_fractions = df.isnull().mean()
    over_threshold = null_fractions[null_fractions > 0.30]
    if not over_threshold.empty:
        for col, frac in over_threshold.items():
            failures.append(f"Column '{col}' is {frac:.1%} null (threshold: 30%).")

    if failures:
        msg = "\n  ".join(failures)
        logger.error("Data validation FAILED:\n  %s", msg)
        raise DataValidationError(
            f"Great Expectations validation failed with {len(failures)} issue(s):\n  {msg}"
        )

    logger.info("✅ All data quality checks passed. Batch is clean.")


# ── Step 4: Save Cumulative Processed History ──────────────────────────────────
def save_cumulative_processed(conn: sqlite3.Connection) -> None:
    """
    Queries all processed records (processed = 1) from the database and persists them
    to data/raw_batch.csv. This ensures the downstream pipeline trains on the full
    historical data rather than just the latest batch.
    """
    query = "SELECT * FROM borrowers WHERE processed = 1"
    df = pd.read_sql_query(query, conn)
    os.makedirs(BATCH_OUTPUT_PATH.parent, exist_ok=True)
    df.to_csv(BATCH_OUTPUT_PATH, index=False)
    logger.info("Saved cumulative processed history (%d rows) to: %s", len(df), BATCH_OUTPUT_PATH)


# ── Main Entry Point ──────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Ingestion — Starting")
    logger.info("=" * 60)

    os.makedirs(DB_PATH.parent, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        # Initialize schema
        create_table_if_not_exists(conn)

        # Seed on first run
        if SEED_TABLE:
            seed_database(conn)

        # Fetch unprocessed batch
        batch_df = fetch_batch(conn)

        # Validate before touching anything downstream
        validate_batch(batch_df)

        # Mark as processed to prevent double-ingestion
        mark_batch_as_processed(conn, batch_df)

        # Save cumulative processed history for process.py
        save_cumulative_processed(conn)

    logger.info("=" * 60)
    logger.info("STAGE 1: Data Ingestion — Complete ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
