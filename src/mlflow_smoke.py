# src/mlflow_smoke.py
import os, yaml, mlflow

# 1) Load config (if present)
cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "train_config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

# 2) Point to your experiment and start a run
mlflow.set_experiment("cmpt2500-project")
with mlflow.start_run(run_name="config-smoke"):
    # 3) Log all config params (safe if empty)
    if isinstance(config, dict):
        mlflow.log_params(config)
    else:
        mlflow.log_param("config_loaded", False)

    # 4) Log a dummy metric so the run is visible
    mlflow.log_metric("smoke_ok", 1.0)

print("✅ Logged config params (if any) and a dummy metric to 'cmpt2500-project'.")