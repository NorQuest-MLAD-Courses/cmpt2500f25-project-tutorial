# Makefile for CMPT 2500 Project - Telecom Churn Prediction

# Configuration - automatically detect best available Python version
PYTHON_EXE = $(shell which python3.12 2>/dev/null || which python3.11 2>/dev/null || which python3.10 2>/dev/null || which python3)
VENV_DIR = .venv
REQUIREMENTS = requirements.txt

# Phony targets don't represent files
.PHONY: setup clean help test test-quick test-cov test-api data data-status
.PHONY: train train-all train-fast mlflow api api-dev api-prep api-test-live
.PHONY: all pipeline

# Default target
help:
	@echo "=== CMPT 2500 Project - Available Targets ==="
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Create virtual environment and install requirements"
	@echo "  clean          - Remove the virtual environment directory"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-quick     - Run tests without slow ones"
	@echo "  test-cov       - Run tests with coverage report"
	@echo "  test-api       - Run only API tests"
	@echo ""
	@echo "Data:"
	@echo "  data           - Pull data from DVC remote"
	@echo "  data-status    - Check DVC data status"
	@echo ""
	@echo "Training:"
	@echo "  train          - Train single model (random_forest)"
	@echo "  train-all      - Train all models with hyperparameter tuning"
	@echo "  train-fast     - Train all models without tuning"
	@echo ""
	@echo "MLflow:"
	@echo "  mlflow         - Start MLflow UI (port 5001)"
	@echo ""
	@echo "API:"
	@echo "  api-prep       - Prepare models for API serving"
	@echo "  api            - Start Flask API (port 5000)"
	@echo "  api-dev        - Start Flask API in dev mode (port 5002)"
	@echo "  api-test-live  - Test live API endpoints (requires API running)"
	@echo ""
	@echo "Workflows:"
	@echo "  all            - Run complete pipeline (data -> train -> mlflow)"
	@echo "  pipeline       - Full pipeline with status messages"
	@echo ""

# Setup target: Create venv and install dependencies
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: $(REQUIREMENTS)
	@echo "Creating virtual environment using $(PYTHON_EXE)..."
	$(PYTHON_EXE) -m venv $(VENV_DIR)
	@echo "Installing packages from $(REQUIREMENTS)..."
	$(VENV_DIR)/bin/python -m pip install --upgrade pip
	$(VENV_DIR)/bin/python -m pip install -r $(REQUIREMENTS)
	@echo ""
	@echo "Setup complete!"
	@echo "Activate the environment using: source $(VENV_DIR)/bin/activate"
	@# On Windows use: .venv\Scripts\activate
	@touch $(VENV_DIR)/bin/activate  # Mark setup as done

# Clean target: Remove the virtual environment
clean:
	@echo "Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo "Clean complete."

# ============ Testing Targets ============

test:
	@echo "Running all tests..."
	pytest tests/ -v

test-quick:
	@echo "Running quick tests (excluding slow ones)..."
	pytest tests/ -v -m "not slow"

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-api:
	@echo "Running API tests only..."
	pytest tests/test_api.py -v

# ============ Data Targets ============

data:
	@echo "Pulling data from DVC remote..."
	dvc pull

data-status:
	@echo "Checking DVC data status..."
	dvc status

# ============ Training Targets ============

train:
	@echo "Training single model (random_forest)..."
	python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --save

train-all:
	@echo "Training all models with hyperparameter tuning..."
	python -m src.train --data data/processed/preprocessed_data.npy --train-all --tune --save

train-fast:
	@echo "Training all models without hyperparameter tuning..."
	python -m src.train --data data/processed/preprocessed_data.npy --train-all --save

# ============ MLflow Targets ============

mlflow:
	@echo "Starting MLflow UI on port 5001..."
	@if [ -f scratch/run_mlflow_ui.sh ]; then \
		./scratch/run_mlflow_ui.sh --port 5001; \
	else \
		mlflow ui --port 5001; \
	fi

# ============ API Targets ============

api-prep:
	@echo "Preparing models for API serving..."
	@if [ -f scratch/prepare_api_models.sh ]; then \
		./scratch/prepare_api_models.sh; \
	else \
		@echo "Error: scratch/prepare_api_models.sh not found"; \
		@echo "Please ensure the scratch directory exists with helper scripts."; \
		exit 1; \
	fi

api:
	@echo "Starting Flask API on port 5000..."
	PORT=5000 python -m src.app

api-dev:
	@echo "Starting Flask API in development mode on port 5002..."
	PORT=5002 python -m src.app

api-test-live:
	@echo "Testing live API endpoints..."
	@if [ -f scratch/test_api.sh ]; then \
		./scratch/test_api.sh --port 5000; \
	else \
		@echo "Error: scratch/test_api.sh not found"; \
		exit 1; \
	fi

# ============ Complete Workflow ============

all: data train-all
	@echo "Pipeline complete! Run 'make mlflow' to view results."

pipeline:
	@echo "=== Running Full Pipeline ==="
	@echo ""
	@echo "Step 1/3: Pulling data from DVC..."
	dvc pull
	@echo ""
	@echo "Step 2/3: Training all models..."
	python -m src.train --data data/processed/preprocessed_data.npy --train-all --save
	@echo ""
	@echo "Step 3/3: Pipeline complete!"
	@echo "To view results, run: make mlflow"
	@echo "To test API, first run: make api-prep && make api"