# Lab 05 Fix Summary

This document summarizes the issues found and fixes applied to make the Lab 05 Prometheus/Grafana monitoring work correctly.

## Issues Found

### Issue 1: Validation Errors Not Counted in Metrics

**Problem**: The `ml_predictions_total` counter was only incremented for:
- Successful predictions (in the `try` block)
- Exceptions during prediction (in the `except` block)

However, validation errors (HTTP 400 responses) returned **before** the try/except block, so they were never counted. This meant:
- Sending invalid data didn't increment error metrics
- The `DemoHighErrorCount` alert could never fire
- Error rate calculations were incorrect

**Location**: `src/app.py`, `make_prediction()` function

**Fix**: Added metric recording for validation errors and missing input:

```python
# For missing input (line 264-272)
if json_data is None:
    prediction_counter.labels(
        model_version=model_version,
        prediction_result='no_input',
        status='error'
    ).inc()
    return {"error": "No input data provided"}, 400

# For validation errors (line 289-297)
error_msg, status_code = validate_input(item)
if error_msg:
    prediction_counter.labels(
        model_version=model_version,
        prediction_result='validation_error',
        status='error'
    ).inc()
    return {"error": error_msg}, status_code
```

### Issue 2: Dashboard JSON Syntax Error

**Problem**: The Grafana dashboard JSON file had a missing comma after the first panel's datasource configuration:

```json
"datasource": {"type": "prometheus", "uid": "PBFA97CFB590B2093"}
"fieldConfig": {
```

This JSON syntax error could cause the dashboard to fail to load.

**Location**: `grafana/dashboards/ml-api-dashboards.json`, line 11

**Fix**: Added missing comma and standardized the datasource UID:

```json
"datasource": {"type": "prometheus", "uid": "prometheus"},
"fieldConfig": {
```

### Issue 3: Inconsistent Datasource UIDs

**Problem**: The first panel used `"uid": "PBFA97CFB590B2093"` while other panels used `"uid": "prometheus"`. This inconsistency could cause some panels to fail to query data.

**Location**: `grafana/dashboards/ml-api-dashboards.json`, line 10

**Fix**: Changed to use the standard `"prometheus"` UID that matches the datasource provisioning configuration.

## Testing Verification

After applying fixes, the following was verified on GitHub Codespaces:

1. **Metrics Recording**: After sending 5 invalid requests:
   ```
   ml_predictions_total{model_version="v1",prediction_result="validation_error",status="error"} 5.0
   ```

2. **Alert Firing**: The `DemoHighErrorCount` alert fired successfully:
   ```json
   {
     "alertname": "DemoHighErrorCount",
     "state": "firing",
     "value": "5e+00"
   }
   ```

3. **Grafana Dashboard**: All panels displaying correctly:
   - API Status: UP
   - Total Predictions: Showing count
   - Error Rate: Calculating correctly
   - Active Alerts: Showing firing alerts

## Files Changed

| File | Change |
|------|--------|
| `src/app.py` | Added error metric recording for validation failures |
| `grafana/dashboards/ml-api-dashboards.json` | Fixed JSON syntax and datasource UID |
| `assignments/Lab 05 - Monitoring and Observability with Prometheus and Grafana.md` | Rewrote with better pedagogical structure |

## Key Learnings

1. **Always record metrics for ALL code paths** - Not just successes and exceptions, but also validation errors and early returns.

2. **Test the complete flow** - The metrics looked correct in code review, but testing revealed they weren't actually being recorded for validation errors.

3. **Check JSON syntax carefully** - A missing comma in a 150-line JSON file is easy to miss but breaks everything.

4. **Use consistent identifiers** - Datasource UIDs, service names, etc. should be consistent across all configuration files.

## Commands Used for Testing

```bash
# Rebuild with code changes
docker-compose up --build -d --force-recreate app

# Generate errors to trigger alert
for i in {1..5}; do
  curl -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}'
done

# Verify metrics are recorded
curl -s http://localhost:5000/metrics | grep ml_predictions_total

# Check alerts are firing
curl -s "http://localhost:9090/api/v1/alerts" | jq '.data.alerts'

# Test successful prediction (note: SeniorCitizen must be "0" or "1")
echo '{"tenure":12,"MonthlyCharges":59.95,"TotalCharges":720.50,"Contract":"One year","PaymentMethod":"Electronic check","OnlineSecurity":"No","TechSupport":"No","InternetService":"DSL","gender":"Female","SeniorCitizen":"0","Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No","PaperlessBilling":"Yes","OnlineBackup":"Yes","DeviceProtection":"No","StreamingTV":"No","StreamingMovies":"No"}' > /tmp/test.json

curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/test.json
```
