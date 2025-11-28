# Lab 05: Instructor Guide - Prometheus and Grafana Monitoring

## Overview

This guide provides step-by-step instructions for demonstrating Prometheus and Grafana monitoring integration in a 2-hour lab session using GitHub Codespaces.

**Target Duration**: 2 hours
**Environment**: GitHub Codespaces
**Prerequisites**: Students completed Lab 04 (Docker containerization is already done)

---

## Pre-Lab Preparation

### Before Class

1. **Verify repository state**:
   - Dockerfile.mlapp and Dockerfile.mlflow exist
   - docker-compose.yml has ML app and MLflow services
   - Models exist in `models/` directory
   - Data exists in `data/processed/` directory

2. **Test Codespaces compatibility**:
   - Docker is available (`docker --version`)
   - Docker Compose is available (`docker-compose --version`)
   - Ports 5000, 5001, 9090, and 3000 can be forwarded

3. **Have these ready**:
   - This instructor guide open
   - Sample prediction JSON for testing
   - The Grafana dashboard JSON (provided in this guide)

---

## Lab Timeline

| Time | Duration | Section | Activity |
|------|----------|---------|----------|
| 0:00 | 10 min | Introduction | Explain Prometheus/Grafana concepts |
| 0:10 | 25 min | Part 1 | Instrument Flask app with Prometheus metrics |
| 0:35 | 15 min | Part 2 | Configure Prometheus |
| 0:50 | 20 min | Part 3 | Configure Grafana |
| 1:10 | 10 min | Part 4 | Update docker-compose.yml |
| 1:20 | 15 min | Part 5 | Build, run, and verify |
| 1:35 | 20 min | Part 6 | Live alert testing and exploration |
| 1:55 | 5 min | Wrap-up | Q&A and assignment explanation |

---

## Opening Codespaces (2 minutes)

1. Navigate to the repository on GitHub
2. Click **Code** > **Codespaces** > **Create codespace on main**
3. Wait for Codespaces to initialize
4. Open a terminal

**Say to students**:
> "Today we're adding monitoring to our containerized ML application. We already have Docker set up from Lab 04. Now we'll add Prometheus for metrics collection and Grafana for visualization."

---

## Section 1: Introduction (10 minutes)

### Key Talking Points

1. **Why monitoring matters for ML systems**:
   > "In production, you can't just deploy and forget. You need visibility into: Is the API responding? How fast? Are predictions failing? Is memory running out?"

2. **Prometheus architecture** (draw or describe):
   - Pull-based: Prometheus *scrapes* metrics from targets
   - Time-series database: Stores metrics with timestamps and labels
   - PromQL: Query language for analysis and alerting

3. **Metric types**:
   - **Counter**: Only goes up (total requests)
   - **Gauge**: Goes up and down (memory usage)
   - **Histogram**: Distribution (latency percentiles)

4. **Grafana's role**:
   > "Grafana is the visualization layer. It queries Prometheus and displays beautiful dashboards."

---

## Section 2: Instrumenting the Flask API (25 minutes)

### Step 2.1: Update requirements.txt

```bash
cd /workspaces/<repo-name>
```

Open `requirements.txt` and add these lines at the end:

```
# Monitoring
prometheus-client==0.21.0
prometheus-flask-exporter==0.23.1
psutil==6.1.0
```

**Explain**:
> "prometheus-client is the official Python library. prometheus-flask-exporter integrates with Flask automatically. psutil lets us monitor CPU and memory."

### Step 2.2: Modify src/app.py

Open `src/app.py` and make these changes:

**Add imports after existing imports (around line 6):**

```python
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import psutil
import threading
import time
```

**Add after `app = Flask(__name__)` (around line 16):**

```python
# --- Prometheus Monitoring ---
# Initialize Prometheus metrics - this automatically exposes /metrics endpoint
metrics = PrometheusMetrics(app)

# Add application info as a metric
metrics.info('app_info', 'ML API Information', version='1.0.0', app_name='churn-prediction-api')

# Custom metrics for ML predictions
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_version', 'prediction_result', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_duration_seconds',
    'Time spent processing prediction requests',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# System resource metrics
memory_usage_gauge = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage_gauge = Gauge('app_cpu_usage_percent', 'CPU usage percentage')
model_loaded_gauge = Gauge('model_loaded', 'Whether models are loaded', ['model_version'])


def monitor_system_resources():
    """Background thread to monitor system resources every 15 seconds."""
    while True:
        try:
            process = psutil.Process(os.getpid())
            memory_usage_gauge.set(process.memory_info().rss)
            cpu_usage_gauge.set(process.cpu_percent(interval=1))
        except Exception:
            pass
        time.sleep(15)
```

**Explain as you type**:
> "Counter for predictions that only goes up. Labels let us slice by model version, result, and status."
> "Histogram for latency with custom buckets - we care about sub-100ms and up to 5 seconds."
> "Gauges for memory and CPU that update in a background thread."

**After successful model loading (around line 54, after the success log), add:**

```python
    # Track model loading status
    model_loaded_gauge.labels(model_version='v1').set(1 if model_v1 else 0)
    model_loaded_gauge.labels(model_version='v2').set(1 if model_v2 else 0)
```

**Modify the `make_prediction` function (around line 201):**

Add `start_time = time.time()` at the very beginning of the function:

```python
def make_prediction(json_data, model, model_version):
    """Shared prediction logic with Prometheus metrics."""
    start_time = time.time()  # ADD THIS LINE
```

**Before the successful return (around line 274), add metrics recording:**

```python
        # Record metrics for successful predictions
        for result in results:
            prediction_counter.labels(
                model_version=model_version,
                prediction_result=result['prediction'],
                status='success'
            ).inc()

        # Record latency
        duration = time.time() - start_time
        prediction_latency.labels(model_version=model_version).observe(duration)

        logger.info(f"{model_version}: Successfully generated {len(results)} prediction(s)")
```

**In the except block (around line 278), add error metrics:**

```python
    except Exception as e:
        # Record failed prediction
        prediction_counter.labels(
            model_version=model_version,
            prediction_result='error',
            status='error'
        ).inc()
        error_msg = f"An error occurred during prediction: {str(e)}"
        logger.error(f"{model_version}: {error_msg}")
        return {"error": error_msg}, 500
```

**Modify the `if __name__ == '__main__':` block at the end:**

```python
if __name__ == '__main__':
    # Start system resource monitoring in background
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()
    logger.info("Started system resource monitoring thread")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
```

### Step 2.3: Verify the Changes Work Locally (Optional)

If you want to quick-test:

```bash
pip install prometheus-client prometheus-flask-exporter psutil
python -m src.app &
curl http://localhost:5000/metrics | head -50
pkill -f "python -m src.app"
```

---

## Section 3: Configure Prometheus (15 minutes)

### Step 3.1: Create Directory Structure

```bash
mkdir -p prometheus/rules
```

### Step 3.2: Create prometheus.yml

Create the file `prometheus/prometheus.yml`:

```yaml
# Prometheus Configuration for ML Application Monitoring

global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Load alerting rules
rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          service: 'prometheus'

  # ML Application API
  - job_name: 'ml-api'
    scrape_interval: 5s
    static_configs:
      - targets: ['app:5000']
        labels:
          service: 'ml-prediction-api'
    metrics_path: /metrics
```

**Explain**:
> "scrape_interval is how often Prometheus pulls metrics. We use 'app:5000' because that's the Docker service name. Prometheus runs inside Docker and talks to other containers by name."

### Step 3.3: Create Alert Rules

Create the file `prometheus/rules/alerts.yml`:

```yaml
groups:
  - name: ml_api_alerts
    rules:
      # Alert: API is down
      - alert: APIDown
        expr: up{job="ml-api"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "ML API is down"
          description: "The ML API has been unreachable for over 30 seconds."

      # Alert: High error rate
      - alert: HighErrorRate
        expr: |
          (sum(rate(ml_predictions_total{status="error"}[2m]))
          / sum(rate(ml_predictions_total[2m]))) > 0.1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate"
          description: "Error rate exceeds 10%"

      # Alert: High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le)) > 0.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "P95 latency exceeds 500ms"

  # Easy-to-trigger alerts for demonstration
  - name: lab_demo_alerts
    rules:
      # Triggers after 3 errors (easy to demo)
      - alert: DemoHighErrorCount
        expr: sum(ml_predictions_total{status="error"}) >= 3
        for: 10s
        labels:
          severity: demo
        annotations:
          summary: "[DEMO] Error threshold reached"
          description: "At least 3 prediction errors have occurred"

      # Triggers with moderate traffic
      - alert: DemoHighRequestRate
        expr: sum(rate(ml_predictions_total[1m])) > 0.5
        for: 30s
        labels:
          severity: demo
        annotations:
          summary: "[DEMO] High request rate detected"
          description: "More than 0.5 requests per second"
```

**Explain**:
> "Alerts have an expression (the condition), a 'for' duration (how long it must be true), labels (for routing), and annotations (human-readable info)."
> "The demo alerts are intentionally easy to trigger so we can see them fire during the lab."

---

## Section 4: Configure Grafana (20 minutes)

### Step 4.1: Create Directory Structure

```bash
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards
```

### Step 4.2: Create Datasource Configuration

Create `grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: "5s"
```

**Explain**:
> "This auto-configures Prometheus as a data source when Grafana starts. No manual UI setup needed."

### Step 4.3: Create Dashboard Provisioning Config

Create `grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'ML Monitoring Dashboards'
    orgId: 1
    folder: 'ML Monitoring'
    folderUid: 'ml-monitoring'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
```

### Step 4.4: Create the Dashboard JSON

Create `grafana/dashboards/ml-api-dashboard.json`:

```json
{
  "annotations": {"list": []},
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "mappings": [
            {"options": {"0": {"color": "red", "text": "DOWN"}}, "type": "value"},
            {"options": {"1": {"color": "green", "text": "UP"}}, "type": "value"}
          ],
          "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "green", "value": 1}]}
        }
      },
      "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
      "id": 1,
      "options": {"colorMode": "background", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "up{job=\"ml-api\"}", "refId": "A"}],
      "title": "API Status",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {"defaults": {"thresholds": {"steps": [{"color": "green", "value": null}]}, "unit": "short"}},
      "gridPos": {"h": 4, "w": 5, "x": 4, "y": 0},
      "id": 2,
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(ml_predictions_total) or vector(0)", "refId": "A"}],
      "title": "Total Predictions",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "thresholds": {"steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 0.05}, {"color": "red", "value": 0.1}]},
          "unit": "percentunit"
        }
      },
      "gridPos": {"h": 4, "w": 5, "x": 9, "y": 0},
      "id": 3,
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(rate(ml_predictions_total{status=\"error\"}[2m])) / sum(rate(ml_predictions_total[2m])) or vector(0)", "refId": "A"}],
      "title": "Error Rate (2m)",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "thresholds": {"steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 0.25}, {"color": "red", "value": 0.5}]},
          "unit": "s"
        }
      },
      "gridPos": {"h": 4, "w": 5, "x": 14, "y": 0},
      "id": 4,
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le)) or vector(0)", "refId": "A"}],
      "title": "P95 Latency",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {"defaults": {"thresholds": {"steps": [{"color": "green", "value": null}]}, "unit": "bytes"}},
      "gridPos": {"h": 4, "w": 5, "x": 19, "y": 0},
      "id": 5,
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "app_memory_usage_bytes or vector(0)", "refId": "A"}],
      "title": "Memory Usage",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "custom": {"drawStyle": "line", "fillOpacity": 20, "lineWidth": 2, "showPoints": "never"},
          "unit": "reqps"
        }
      },
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
      "id": 6,
      "options": {"legend": {"calcs": ["mean", "max", "last"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "sum(rate(ml_predictions_total[1m])) by (model_version)", "legendFormat": "{{model_version}}", "refId": "A"},
        {"expr": "sum(rate(ml_predictions_total[1m]))", "legendFormat": "Total", "refId": "B"}
      ],
      "title": "Request Rate by Model Version",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "custom": {"drawStyle": "line", "fillOpacity": 20, "lineWidth": 2},
          "thresholds": {"steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 0.25}, {"color": "red", "value": 0.5}]},
          "unit": "s"
        }
      },
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
      "id": 7,
      "options": {"legend": {"calcs": ["mean", "max", "last"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le))", "legendFormat": "p50", "refId": "A"},
        {"expr": "histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le))", "legendFormat": "p95", "refId": "B"},
        {"expr": "histogram_quantile(0.99, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le))", "legendFormat": "p99", "refId": "C"}
      ],
      "title": "Prediction Latency Percentiles",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "gridPos": {"h": 8, "w": 8, "x": 0, "y": 12},
      "id": 8,
      "options": {"legend": {"displayMode": "table", "placement": "right", "values": ["value", "percent"]}, "pieType": "pie"},
      "targets": [{"expr": "sum(ml_predictions_total) by (prediction_result)", "legendFormat": "{{prediction_result}}", "refId": "A"}],
      "title": "Predictions by Result",
      "type": "piechart"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "gridPos": {"h": 8, "w": 8, "x": 8, "y": 12},
      "id": 9,
      "options": {"legend": {"displayMode": "table", "placement": "right", "values": ["value", "percent"]}, "pieType": "pie"},
      "targets": [{"expr": "sum(ml_predictions_total) by (status)", "legendFormat": "{{status}}", "refId": "A"}],
      "title": "Predictions by Status",
      "type": "piechart"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "gridPos": {"h": 8, "w": 8, "x": 16, "y": 12},
      "id": 10,
      "options": {"alertInstanceLabelFilter": "", "dashboardAlerts": false, "maxItems": 20, "sortOrder": 1, "stateFilter": {"error": true, "firing": true, "noData": false, "normal": false, "pending": true}},
      "title": "Active Alerts",
      "type": "alertlist"
    }
  ],
  "refresh": "5s",
  "schemaVersion": 38,
  "tags": ["ml", "api", "monitoring"],
  "templating": {"list": []},
  "time": {"from": "now-15m", "to": "now"},
  "timepicker": {},
  "timezone": "browser",
  "title": "ML API Monitoring Dashboard",
  "uid": "ml-api-monitoring",
  "version": 1
}
```

---

## Section 5: Update Docker Compose (10 minutes)

### Step 5.1: Add Prometheus and Grafana Services

Open `docker-compose.yml` and add the following services after `mlflow`:

```yaml
  # Prometheus - Metrics Collection
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - ml-network
    depends_on:
      - app
    restart: unless-stopped

  # Grafana - Metrics Visualization
  grafana:
    image: grafana/grafana:10.1.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    networks:
      - ml-network
    depends_on:
      - prometheus
    restart: unless-stopped
```

### Step 5.2: Add Volumes

Add to the `volumes:` section at the bottom:

```yaml
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
```

---

## Section 6: Build, Run, and Verify (15 minutes)

### Step 6.1: Build and Start Everything

```bash
docker-compose up --build -d
```

**Explain while building**:
> "Docker is rebuilding the app image with our new monitoring dependencies. Prometheus and Grafana are using official images, so they just download."

### Step 6.2: Check Container Status

```bash
docker-compose ps
```

All containers should show "Up" status.

### Step 6.3: Access Services

In Codespaces, go to the **PORTS** tab and open:

| Port | Service | What to Check |
|------|---------|---------------|
| 5000 | ML API | `/health` returns ok, `/metrics` shows Prometheus metrics |
| 5001 | MLflow | UI loads |
| 9090 | Prometheus | Status > Targets shows both targets as "UP" |
| 3000 | Grafana | Login with admin/admin, navigate to Dashboards > ML Monitoring |

### Step 6.4: Verify Prometheus Targets

Navigate to Prometheus (port 9090) > Status > Targets

**Both targets should show "UP" in green**:
- `prometheus` (self-monitoring)
- `ml-api` (your application)

### Step 6.5: Generate Initial Traffic

```bash
# Test the health endpoint
curl http://localhost:5000/health

# Make a successful prediction
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 59.95,
    "TotalCharges": 720.50,
    "Contract": "One year",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "InternetService": "DSL",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "PaperlessBilling": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
  }'
```

---

## Section 7: Live Alert Testing (20 minutes)

### Step 7.1: Explore Prometheus Queries

Navigate to Prometheus (port 9090) and demonstrate these queries:

```promql
# Check API is up
up{job="ml-api"}

# View all prediction metrics
ml_predictions_total

# Calculate request rate
rate(ml_predictions_total[1m])

# View by model version
sum by (model_version) (ml_predictions_total)

# Calculate average latency
rate(ml_prediction_duration_seconds_sum[5m]) / rate(ml_prediction_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le))
```

### Step 7.2: View Grafana Dashboard

1. Navigate to Grafana (port 3000)
2. Login with admin/admin
3. Go to Dashboards > ML Monitoring > ML API Monitoring Dashboard
4. Point out each panel and what it shows

### Step 7.3: Trigger the Demo Alerts

**Goal**: Trigger `DemoHighErrorCount` by sending invalid requests

```bash
# Send 5 invalid requests to trigger the error count alert
for i in {1..5}; do
  curl -s -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}'
  echo ""
done
```

**Now check alerts**:
1. Go to Prometheus > Alerts
2. `DemoHighErrorCount` should show as "Pending" then "Firing"

**Explain**:
> "The alert is pending because it needs to be true for the 'for' duration (10 seconds). Once that passes, it fires."

### Step 7.4: Trigger High Request Rate Alert

```bash
# Generate traffic to trigger the request rate alert
for i in {1..30}; do
  curl -s -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"tenure":12,"MonthlyCharges":59.95,"TotalCharges":720.50,"Contract":"One year","PaymentMethod":"Electronic check","OnlineSecurity":"No","TechSupport":"No","InternetService":"DSL","gender":"Female","SeniorCitizen":"No","Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No","PaperlessBilling":"Yes","OnlineBackup":"Yes","DeviceProtection":"No","StreamingTV":"No","StreamingMovies":"No"}' &
done
wait
echo "Sent 30 requests"
```

Check Prometheus Alerts - `DemoHighRequestRate` should fire.

### Step 7.5: View Alerts in Grafana

1. Go back to the Grafana dashboard
2. The "Active Alerts" panel should show the firing alerts
3. Watch the request rate and error rate graphs update

### Step 7.6: (Optional) Trigger APIDown Alert

```bash
# Stop the app container
docker-compose stop app

# Wait 30+ seconds, then check Prometheus Alerts
# APIDown should fire

# Restart the app
docker-compose start app
```

---

## Section 8: Wrap-up (5 minutes)

### Key Takeaways

1. **Prometheus collects metrics** via a pull model from `/metrics` endpoint
2. **prometheus-flask-exporter** makes instrumentation easy
3. **Grafana visualizes** the data with dashboards
4. **Alerting** proactively notifies you of issues
5. **Docker Compose** orchestrates the entire monitoring stack

### Assignment Reminder

Students need to:
- Implement this in their own projects
- Adapt metrics to their specific ML use case
- Submit screenshots of working dashboards AND firing alerts
- Write a brief report on their monitoring strategy

---

## Quick Reference Commands

```bash
# Start everything
docker-compose up --build -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f app

# Stop everything
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart a service
docker-compose restart app

# Check status
docker-compose ps

# Execute command in container
docker-compose exec app bash
```

---

## Troubleshooting

### Issue: Prometheus target shows "DOWN"

```bash
# Check if app is running
docker-compose ps

# Check app logs
docker-compose logs app

# Test metrics endpoint from inside Docker network
docker-compose exec prometheus wget -qO- http://app:5000/metrics | head
```

### Issue: Grafana shows "No data"

1. Verify Prometheus data source (Settings > Data Sources > Prometheus > Test)
2. Check Prometheus has data (query `up` in Prometheus UI)
3. Make sure some predictions have been made
4. Check dashboard time range (set to "Last 15 minutes")

### Issue: Alerts not firing

1. Go to Prometheus > Alerts to see rule evaluation
2. Check if the expression is returning expected values
3. Verify the `for` duration has passed
4. Check Prometheus > Status > Rules for any errors

---

## Files Created in This Lab

```
prometheus/
  prometheus.yml
  rules/
    alerts.yml
grafana/
  provisioning/
    datasources/
      prometheus.yml
    dashboards/
      dashboards.yml
  dashboards/
    ml-api-dashboard.json
```

**Modified Files**:
- `requirements.txt` - added monitoring dependencies
- `src/app.py` - added Prometheus instrumentation
- `docker-compose.yml` - added Prometheus and Grafana services
