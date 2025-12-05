# Lab 05 Fix Summary

This document summarizes the issues found and fixes applied to make Lab 05 (Prometheus/Grafana monitoring) work correctly.

---

## Issues Found and Fixed

### Issue 1: Validation Errors Not Counted in Metrics (Critical)

**Problem**: The `ml_predictions_total` counter was only incremented for:
- Successful predictions (in the `try` block)
- Exceptions during prediction (in the `except` block)

Validation errors (HTTP 400 responses) returned **before** the try/except block, so they were never counted. This caused:
- Sending invalid data didn't increment error metrics
- The `DemoHighErrorCount` alert could never fire
- Error rate calculations were incorrect

**Location**: `src/app.py`, `make_prediction()` function

**Fix**: Added metric recording for all error paths:

```python
# For missing input
if json_data is None:
    prediction_counter.labels(
        model_version=model_version,
        prediction_result='no_input',
        status='error'
    ).inc()
    return {"error": "No input data provided"}, 400

# For validation errors
if error_msg:
    prediction_counter.labels(
        model_version=model_version,
        prediction_result='validation_error',
        status='error'
    ).inc()
    return {"error": error_msg}, status_code
```

### Issue 2: Dashboard JSON Syntax Error

**Problem**: Missing comma in `grafana/dashboards/ml-api-dashboards.json` line 11, and inconsistent datasource UID (`"PBFA97CFB590B2093"` vs `"prometheus"`).

**Fix**: Added missing comma and standardized UID to `"prometheus"`.

### Issue 3: Missing HighLatency Alert

**Problem**: The `HighLatency` alert was documented but not in the actual `alerts.yml` file.

**Fix**: Added the alert to `prometheus/rules/alerts.yml`:

```yaml
- alert: HighLatency
  expr: |
    histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le)) > 0.5
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "High prediction latency"
    description: "95th percentile latency exceeds 500ms."
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/app.py` | Added error metric recording for validation failures and missing input |
| `grafana/dashboards/ml-api-dashboards.json` | Fixed JSON syntax and datasource UID |
| `prometheus/rules/alerts.yml` | Added HighLatency alert, fixed descriptions |
| `assignments/Lab 05 - *.md` | Rewrote documentation with better structure |

---

## Verification

After fixes, all components work correctly:

1. **Metrics**: All error types recorded
   ```
   ml_predictions_total{status="error", prediction_result="validation_error"} 5.0
   ```

2. **Alerts**: `DemoHighErrorCount` fires after 3+ errors

3. **Dashboard**: All panels display data correctly

---

## Key Learning

**Always record metrics for ALL code paths** - not just successes and exceptions, but also validation errors and early returns. If a code path can fail, it should increment the error counter.
