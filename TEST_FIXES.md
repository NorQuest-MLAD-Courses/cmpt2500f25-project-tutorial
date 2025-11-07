# Test Fixes - DataFrame dtype and Port Configuration

## Issues Fixed

### Issue 1: DataFrame dtype Error with SeniorCitizen

**Error Message:**
```
ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
```

**Root Cause:**
After the `normalize_senior_citizen()` function converts SeniorCitizen from string to integer, pandas DataFrame creation was having trouble inferring proper dtypes for columns with mixed types.

**Fix Applied:**
Modified `src/app.py` (lines 313-326) to explicitly set dtypes after DataFrame creation:

```python
# Ensure proper dtypes to avoid pandas inference issues
# SeniorCitizen must be int (it's categorical but stored as 0/1)
if 'SeniorCitizen' in input_df.columns:
    input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)

# Ensure numerical features are numeric types
for col in NUMERICAL_FEATURES:
    if col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

# Ensure categorical features (except SeniorCitizen) are strings
for col in CATEGORICAL_FEATURES:
    if col in input_df.columns and col != 'SeniorCitizen':
        input_df[col] = input_df[col].astype(str)
```

This ensures:
- SeniorCitizen is always int (0 or 1)
- Numerical features are properly typed as numeric
- Categorical features are strings
- No type inference issues in pandas

---

### Issue 2: Docker Test Port Hardcoding

**Error:**
Docker tests were hardcoded to use `localhost:5000` and `localhost:5001`, but many users need to use different ports due to port conflicts (e.g., macOS AirPlay Receiver).

**Fix Applied:**
Modified `tests/test_docker.py` to support port configuration via environment variables:

```python
# Configuration at top of file (lines 18-25)
API_PORT = os.getenv("API_PORT", "5000")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5001")

API_BASE_URL = f"http://localhost:{API_PORT}"
MLFLOW_BASE_URL = f"http://localhost:{MLFLOW_PORT}"
```

All test functions now use `API_BASE_URL` and `MLFLOW_BASE_URL` instead of hardcoded URLs.

---

## How to Test the Fixes

### Step 1: Rebuild Docker Containers

Stop existing containers and rebuild with updated code:

```bash
docker-compose down
docker-compose up -d --build
```

If you're using a custom port (e.g., 5002), your `docker-compose.yml` should have:

```yaml
ml-app:
  ports:
    - "5002:5000"  # Host port 5002, container port 5000
```

### Step 2: Verify Containers Are Running

```bash
docker-compose ps
```

Expected output:
```
NAME            STATE    PORTS
churn-api       Up       0.0.0.0:5002->5000/tcp  # Or 5000 if using default
mlflow-server   Up       0.0.0.0:5001->5001/tcp
```

### Step 3: Test API with Both SeniorCitizen Formats

**Test with integer format (0):**
```bash
curl -X POST http://localhost:5002/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12, "MonthlyCharges": 59.95, "TotalCharges": 720.50,
    "Contract": "One year", "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No", "TechSupport": "No", "InternetService": "DSL",
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "PaperlessBilling": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "StreamingTV": "No", "StreamingMovies": "No"
  }'
```

**Expected:** ✅ Success with prediction (no dtype error)

**Test with string format ("No"):**
```bash
curl -X POST http://localhost:5002/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12, "MonthlyCharges": 59.95, "TotalCharges": 720.50,
    "Contract": "One year", "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No", "TechSupport": "No", "InternetService": "DSL",
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "PaperlessBilling": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "StreamingTV": "No", "StreamingMovies": "No"
  }'
```

**Expected:** ✅ Success with prediction (normalized to 0)

### Step 4: Run API Tests (Non-Docker)

If you have pytest installed locally:

```bash
# Install dependencies if needed
pip install -r requirements.txt

# Run API tests
pytest tests/test_api.py -v
```

**Expected:** All 6 previously failing tests should now pass:
- `test_v1_predict_single`
- `test_v1_predict_batch`
- `test_v2_predict_single`
- `test_v2_predict_batch`
- `test_predict_senior_citizen_integer`
- `test_predict_senior_citizen_string`

### Step 5: Run Docker Tests with Custom Port

If your containers are running on a non-standard port (e.g., 5002), set the environment variable:

```bash
# For port 5002
export API_PORT=5002
export MLFLOW_PORT=5001

# Run Docker tests
pytest tests/test_docker.py -v
```

Or in a single command:

```bash
API_PORT=5002 MLFLOW_PORT=5001 pytest tests/test_docker.py -v
```

**Expected:** All 11 previously failing Docker tests should now pass:
- `test_docker_images_exist`
- `test_containers_are_running`
- `test_api_health_endpoint`
- `test_api_home_endpoint`
- `test_api_prediction_v1`
- `test_api_prediction_v2`
- `test_api_validation_error`
- `test_mlflow_ui_accessible`
- `test_docker_network_communication`
- `test_docker_volumes_mounted`
- `test_container_health_checks`

### Step 6: Run Full Test Suite

```bash
# If using custom ports
API_PORT=5002 MLFLOW_PORT=5001 pytest tests/ -v

# If using default ports
pytest tests/ -v
```

**Expected:** All tests should pass (previously 152 passed + 17 fixed = 169 total)

---

## Files Modified

1. **`src/app.py`** (lines 308-332)
   - Added explicit dtype setting after DataFrame creation
   - Prevents pandas type inference errors
   - No breaking changes to API

2. **`tests/test_docker.py`** (lines 18-25, and all test functions)
   - Added `API_PORT` and `MLFLOW_PORT` configuration
   - Replaced hardcoded URLs with `API_BASE_URL` and `MLFLOW_BASE_URL`
   - Backward compatible (defaults to standard ports 5000/5001)

---

## Troubleshooting

### If API Tests Still Fail

1. **Check Python version**: Requires Python 3.12 (or compatible pandas/numpy versions)
   ```bash
   python --version
   ```

2. **Verify dependencies are installed:**
   ```bash
   pip list | grep -E "pandas|numpy|scikit-learn"
   ```

3. **Check that preprocessing pipeline exists:**
   ```bash
   ls -la data/processed/preprocessing_pipeline.pkl
   ```

4. **Pull DVC data if missing:**
   ```bash
   dvc pull
   ```

### If Docker Tests Still Fail

1. **Verify correct port in docker-compose.yml:**
   ```bash
   grep -A 2 "ports:" docker-compose.yml
   ```

2. **Check container logs:**
   ```bash
   docker-compose logs ml-app
   ```

3. **Verify port environment variable:**
   ```bash
   echo $API_PORT
   # Should output: 5002 (or your custom port)
   ```

4. **Test API manually first:**
   ```bash
   curl http://localhost:5002/health
   # Should return: {"status": "ok"}
   ```

---

## What Changed vs. What Didn't

### ✅ Changed:
- DataFrame creation now explicitly sets dtypes
- Docker tests support custom ports via environment variables
- More robust handling of mixed-type data

### ✅ Didn't Change (No Breaking Changes):
- API endpoints remain the same
- Model files unchanged
- Preprocessing pipeline unchanged
- DVC data unchanged
- Docker configuration unchanged
- SeniorCitizen normalization logic unchanged (still accepts both formats)

---

## Summary

Both issues are now fixed:
1. **DataFrame dtype error** - Resolved by explicit type casting
2. **Port hardcoding in tests** - Resolved by environment variable configuration

All tests should pass when run with appropriate environment configuration.
