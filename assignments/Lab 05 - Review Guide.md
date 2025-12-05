# Lab 05 Review Guide (10 Minutes)

Use this guide to walk through Lab 05 with students. This covers what to show, what to say, and addresses the bug that prevented the alert demo in the previous session.

---

## Opening (1 minute)

**Say**: "In the last session, we set up Prometheus and Grafana for monitoring our ML API. Everything worked except the alerts - when we sent invalid requests, the alerts didn't fire. Today I'll show you the working alerts and explain what the bug was."

---

## Part 1: Show the Working System (3 minutes)

### Start the services (if not already running)

```bash
docker-compose up -d
docker-compose ps
```

**Show**: All 4 containers running (ml-api, mlflow, prometheus, grafana)

### Show Prometheus Targets

**URL**: http://localhost:9090/targets

**Say**: "Prometheus scrapes metrics from our API every 5 seconds. Both targets show UP - Prometheus itself and our ML API."

### Show Grafana Dashboard

**URL**: http://localhost:3000 (admin/admin)
Navigate to: Dashboards > ML Monitoring > ML API Monitoring Dashboard

**Say**: "Grafana visualizes the metrics Prometheus collects. We have:
- API status (UP/DOWN)
- Total predictions count
- Error rate and latency
- Active alerts panel (bottom right)"

---

## Part 2: Demonstrate Alerts Working (4 minutes)

### Generate Errors to Trigger Alert

```bash
# Send 5 invalid requests
for i in {1..5}; do
  curl -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}'
done
```

**Say**: "I'm sending invalid data to trigger our DemoHighErrorCount alert, which fires when we have 3 or more errors."

### Show Alert Firing

**URL**: http://localhost:9090/alerts

**Say**: "After about 10 seconds, you'll see DemoHighErrorCount change from 'inactive' to 'pending' to 'firing'. The `for: 10s` means it waits 10 seconds before firing to avoid false alarms from temporary spikes."

**Show**: The alert in "firing" state (red)

### Show in Grafana

**Say**: "The same alert also appears in our Grafana dashboard's Active Alerts panel."

---

## Part 3: Explain the Bug (2 minutes)

**Say**: "Last time, the alerts wouldn't fire. Here's why."

### The Problem

**Say**: "Our prediction function had three outcomes:
1. Success - metrics recorded ✓
2. Exception during prediction - metrics recorded ✓
3. Validation error (invalid input) - metrics NOT recorded ✗

When we sent invalid data, the code returned a 400 error BEFORE reaching the metrics code. So Prometheus never saw any errors, and the alert couldn't fire."

### Show the code (optional)

```python
# OLD CODE - validation errors returned early, no metrics
if error_msg:
    return {"error": error_msg}, status_code  # <- No counter increment!

# FIXED CODE - now we record the error first
if error_msg:
    prediction_counter.labels(
        model_version=model_version,
        prediction_result='validation_error',
        status='error'
    ).inc()  # <- Now we count it!
    return {"error": error_msg}, status_code
```

**Say**: "The lesson: Always record metrics for ALL code paths - successes, exceptions, AND validation errors. If something can fail, count it."

---

## Wrap Up (30 seconds)

**Say**: "So now our monitoring stack is complete:
- Prometheus collects metrics every 5 seconds
- We track predictions, errors, and latency
- Alerts fire when thresholds are exceeded
- Grafana shows everything in one dashboard

Questions?"

---

## Quick Commands Reference

```bash
# Start everything
docker-compose up -d

# Check services
docker-compose ps

# Generate successful traffic
echo '{"tenure":12,"MonthlyCharges":59.95,"TotalCharges":720.50,"Contract":"One year","PaymentMethod":"Electronic check","OnlineSecurity":"No","TechSupport":"No","InternetService":"DSL","gender":"Female","SeniorCitizen":"0","Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No","PaperlessBilling":"Yes","OnlineBackup":"Yes","DeviceProtection":"No","StreamingTV":"No","StreamingMovies":"No"}' > /tmp/test.json

for i in {1..10}; do
  curl -s -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" -d @/tmp/test.json
done

# Generate errors (to trigger alert)
for i in {1..5}; do
  curl -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}'
done

# Check alerts via API
curl -s "http://localhost:9090/api/v1/alerts" | jq '.data.alerts'

# Check metrics
curl -s http://localhost:5000/metrics | grep ml_predictions_total
```

---

## URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana Dashboard | http://localhost:3000 | admin/admin |
| Prometheus Alerts | http://localhost:9090/alerts | - |
| Prometheus Targets | http://localhost:9090/targets | - |
| ML API Metrics | http://localhost:5000/metrics | - |
