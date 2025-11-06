"""
Tests package for the Telecom Churn Prediction project.

This package contains automated tests using pytest to ensure the functionality
and correctness of the source code modules (preprocessing, training, evaluation,
prediction, and API).

Test Organization:
------------------
- test_preprocess.py: Unit tests for data preprocessing pipeline
- test_train.py: Unit tests for model training and MLflow logging
- test_evaluate.py: Unit tests for model evaluation and metrics
- test_predict.py: Unit tests for making predictions with trained models
- test_api.py: Unit tests for Flask REST API endpoints
- test_integration.py: Integration tests for end-to-end workflows
- conftest.py: Shared pytest fixtures (e.g., cleanup_mlflow)

Running Tests:
--------------
- All tests: `pytest` or `make test`
- With coverage: `pytest --cov=src --cov-report=html` or `make test-cov`
- Only API tests: `pytest tests/test_api.py` or `make test-api`
- Quick tests (no slow): `pytest -m "not slow"` or `make test-quick`

Available Fixtures (from conftest.py):
--------------------------------------
- cleanup_mlflow: Auto-cleanup MLflow runs between tests (autouse=True)

Test Markers:
-------------
- @pytest.mark.unit: Unit tests for individual functions
- @pytest.mark.integration: Integration tests for workflows
- @pytest.mark.slow: Tests that take longer to run
- @pytest.mark.cli: Tests for command-line interfaces
- @pytest.mark.skip_ci: Skip in CI/CD pipeline
"""