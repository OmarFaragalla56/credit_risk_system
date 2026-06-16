# ==============================================================================
# Credit Risk MLOps Pipeline — Makefile
# ==============================================================================
# Usage:
#   make ingest    → Pull & validate new data from SQLite warehouse
#   make process   → Feature engineering + pipeline persistence
#   make train     → Optuna hyperparameter search + MLflow logging
#   make promote   → Compare champion vs. challenger; promote if better
#   make pipeline  → Run all 4 stages sequentially (full pipeline)
#   make test      → Run pytest suite
#   make mlflow    → Launch MLflow tracking UI in browser
#   make clean     → Remove processed data and challenger artifacts
# ==============================================================================

.PHONY: ingest process train promote pipeline test mlflow clean

# Default Python interpreter (uses the active virtual environment)
PYTHON := python

# ── Individual Pipeline Stages ──────────────────────────────────────────────
ingest:
	@echo "========================================"
	@echo " STAGE 1: Data Ingestion"
	@echo "========================================"
	$(PYTHON) -m src.ingest

process:
	@echo "========================================"
	@echo " STAGE 2: Data Processing"
	@echo "========================================"
	$(PYTHON) -m src.process

train:
	@echo "========================================"
	@echo " STAGE 3: Model Training (Optuna + MLflow)"
	@echo "========================================"
	$(PYTHON) -m src.train

promote:
	@echo "========================================"
	@echo " STAGE 4: Model Evaluation & Promotion"
	@echo "========================================"
	$(PYTHON) -m src.evaluate

# ── Full Pipeline ────────────────────────────────────────────────────────────
pipeline: ingest process train promote
	@echo ""
	@echo "========================================"
	@echo " ✅ Full MLOps Pipeline Complete"
	@echo "========================================"

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	@echo "========================================"
	@echo " Running Pytest Suite"
	@echo "========================================"
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── MLflow UI ────────────────────────────────────────────────────────────────
mlflow:
	@echo "Launching MLflow UI at http://localhost:5000 ..."
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning up processed data and challenger artifacts..."
	rm -f data/raw_batch.csv
	rm -f data/processed/train_processed.csv
	rm -f data/processed/test_processed.csv
	rm -f models/challenger_xgboost.pkl
	@echo "Clean complete."
