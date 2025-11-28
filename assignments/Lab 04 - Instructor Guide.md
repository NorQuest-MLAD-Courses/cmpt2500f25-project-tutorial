# Lab 04: Instructor Guide - Prometheus and Grafana Monitoring

## Overview

This guide provides step-by-step instructions for demonstrating Prometheus and Grafana monitoring integration in a 2-hour lab session using GitHub Codespaces.

**Target Duration**: 2 hours
**Environment**: GitHub Codespaces
**Prerequisites**: Students should have completed Labs 01-03

---

## Pre-Lab Preparation (Before Class)

### 1. Verify Repository State

Ensure the repository is at the correct stage:
- Flask API is functional (`src/app.py`)
- Models exist in `models/` directory
- Data exists in `data/processed/` directory

### 2. Test Codespaces Compatibility

Before the lab, verify:
- Docker is available in Codespaces (`docker --version`)
- Docker Compose is available (`docker-compose --version`)
- Port forwarding works for ports 5000, 9090, and 3000

### 3. Prepare Files in Advance (Optional)

To save time, you can have the monitoring files ready in a separate branch or gist that students can reference.

---

## Lab Timeline

| Time | Section | Activity |
|------|---------|----------|
| 0:00-0:10 | Introduction | Explain Prometheus/Grafana concepts |
| 0:10-0:35 | Part 1 | Add Prometheus metrics to Flask app |
| 0:35-0:55 | Part 2 | Create Dockerfile and docker-compose.yml |
| 0:55-1:10 | Part 3 | Configure Prometheus |
| 1:10-1:35 | Part 4 | Set up Grafana |
| 1:35-1:55 | Part 5 | Run, test, and explore metrics |
| 1:55-2:00 | Wrap-up | Q&A and assignment explanation |

---

## Step-by-Step Demonstration

### Opening Codespaces (5 minutes)

1. Navigate to the repository on GitHub
2. Click **Code** > **Codespaces** > **Create codespace on main** (or the appropriate branch)
3. Wait for Codespaces to initialize
4. Open a terminal in Codespaces

**Say to students**: "We'll be working entirely in GitHub Codespaces today. This gives us a consistent environment with Docker pre-installed."

### Section 1: Introduction (10 minutes)

**Talking Points**:

1. **Why monitoring matters for ML systems**:
   - "In production, you can't just deploy a model and forget about it"
   - "You need to know: Is the API responsive? Are predictions accurate? Is the system healthy?"

2. **Prometheus architecture** (draw on whiteboard or show diagram):
   - Pull-based: Prometheus scrapes metrics from targets
   - Time-series database: Stores metrics with timestamps
   - PromQL: Query language for analysis

3. **Grafana's role**:
   - Visualization layer on top of Prometheus
   - Dashboards, alerts, annotations

---

### Section 2: Instrumenting the Flask API (25 minutes)

#### Step 2.1: Update requirements.txt

```bash
# In the terminal
cd /workspaces/<repo-name>
```

**Type/demonstrate**:
```bash
# Open requirements.txt and add these lines
echo "
# Monitoring
prometheus-client==0.21.0
prometheus-flask-exporter==0.23.1
psutil==6.1.0" >> requirements.txt
```

Or edit the file directly to add:
```
prometheus-client==0.21.0
prometheus-flask-exporter==0.23.1
psutil==6.1.0
```

**Explain**: "prometheus-client is the official Python client. prometheus-flask-exporter makes it easy to add metrics to Flask apps. psutil lets us monitor system resources."

#### Step 2.2: Modify src/app.py

Open `src/app.py` and make the following changes:

**Add imports at the top** (after existing imports):

```python
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import psutil
import threading
import time
```

**Add after `app = Flask(__name__)`**:

```python
# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Add application info
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

# System metrics
memory_usage_gauge = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage_gauge = Gauge('app_cpu_usage_percent', 'CPU usage percentage')
model_loaded_gauge = Gauge('model_loaded', 'Whether models are loaded', ['model_version'])
```

**Add the resource monitoring function** (before the routes):

```python
def monitor_system_resources():
    """Background thread to monitor system resources every 15 seconds."""
    while True:
        try:
            process = psutil.Process(os.getpid())
            memory_usage_gauge.set(process.memory_info().rss)
            cpu_usage_gauge.set(process.cpu_percent(interval=1))
        except Exception:
            pass  # Ignore errors in background monitoring
        time.sleep(15)
```

**Modify the model loading section** (around line 48-63) to track model loading:

After the models are loaded successfully, add:
```python
    # Track model loading status
    model_loaded_gauge.labels(model_version='v1').set(1 if model_v1 else 0)
    model_loaded_gauge.labels(model_version='v2').set(1 if model_v2 else 0)
```

**Modify the `make_prediction` function** to add metrics tracking:

Find the `make_prediction` function and add timing at the start:
```python
def make_prediction(json_data, model, model_version):
    """Shared prediction logic with Prometheus metrics."""
    start_time = time.time()
```

After a successful prediction (before the return statement at the end of try block):
```python
        # Record metrics
        for result in results:
            prediction_counter.labels(
                model_version=model_version,
                prediction_result=result['prediction'],
                status='success'
            ).inc()

        duration = time.time() - start_time
        prediction_latency.labels(model_version=model_version).observe(duration)
```

In the except block, add:
```python
        prediction_counter.labels(
            model_version=model_version,
            prediction_result='error',
            status='error'
        ).inc()
```

**Modify the `if __name__ == '__main__':` block**:

```python
if __name__ == '__main__':
    # Start system resource monitoring in background
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
```

**Explain as you go**:
- "Counter is for things that only go up - like total requests"
- "Histogram lets us track distributions - perfect for latency"
- "Gauge is for values that go up and down - like memory usage"
- "Labels let us slice and dice the data - by model version, status, etc."

#### Step 2.3: Verify the metrics endpoint works

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (if models are available)
python -m src.app &

# In another terminal, check metrics
curl http://localhost:5000/metrics

# Stop the app
pkill -f "python -m src.app"
```

**Show students**: "Look, we now have a /metrics endpoint that Prometheus can scrape!"

---

### Section 3: Docker Configuration (20 minutes)

#### Step 3.1: Create Dockerfile

```bash
touch Dockerfile
```

**Content**:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "-m", "src.app"]
```

**Explain**:
- "We use python:3.12-slim for a smaller image"
- "COPY requirements.txt first for better layer caching"
- "HEALTHCHECK lets Docker know if our container is healthy"

#### Step 3.2: Create docker-compose.yml

```bash
touch docker-compose.yml
```

**Content**:

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: ml-api
    ports:
      - "5000:5000"
    environment:
      - PORT=5000
    networks:
      - monitoring-network
    restart: unless-stopped

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
      - monitoring-network
    restart: unless-stopped

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
    depends_on:
      - prometheus
    networks:
      - monitoring-network
    restart: unless-stopped

networks:
  monitoring-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

**Explain**:
- "Three services: our app, Prometheus, and Grafana"
- "They all share a network so they can communicate"
- "Volumes persist data across container restarts"
- "We mount configuration files as read-only for security"

#### Step 3.3: Create .dockerignore

```bash
touch .dockerignore
```

**Content**:

```
.git
.gitignore
.dvc
*.pyc
__pycache__
.pytest_cache
.venv
venv
*.egg-info
.coverage
htmlcov
outputs/
notebooks/
tests/
*.md
Makefile
```

---

### Section 4: Prometheus Configuration (15 minutes)

#### Step 4.1: Create directory structure

```bash
mkdir -p prometheus/rules
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards
```

#### Step 4.2: Create prometheus.yml

```bash
touch prometheus/prometheus.yml
```

**Content**:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          service: 'prometheus'

  - job_name: 'ml-api'
    scrape_interval: 5s
    static_configs:
      - targets: ['app:5000']
        labels:
          service: 'ml-prediction-api'
    metrics_path: /metrics
```

**Explain**:
- "scrape_interval: how often Prometheus pulls metrics"
- "We use 'app:5000' because that's the service name in Docker"
- "labels help us filter and group metrics later"

#### Step 4.3: Create alert rules (optional, for demonstration)

```bash
touch prometheus/rules/alerts.yml
```

**Content**:

```yaml
groups:
  - name: ml_api_alerts
    rules:
      - alert: APIHighErrorRate
        expr: |
          (
            sum(rate(ml_predictions_total{status="error"}[5m]))
            /
            sum(rate(ml_predictions_total[5m]))
          ) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in ML API"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"

      - alert: APIHighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le)
          ) > 1.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "95th percentile latency is {{ $value }}s"

      - alert: APIDown
        expr: up{job="ml-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML API is down"
          description: "The ML API has been unreachable for over 1 minute"
```

**Explain**:
- "Alerts fire when conditions are true for a duration"
- "We use PromQL expressions to define conditions"
- "Labels like severity help with alert routing"

---

### Section 5: Grafana Configuration (25 minutes)

#### Step 5.1: Create datasource configuration

```bash
touch grafana/provisioning/datasources/prometheus.yml
```

**Content**:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

#### Step 5.2: Create dashboard provisioning config

```bash
touch grafana/provisioning/dashboards/dashboards.yml
```

**Content**:

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
      foldersFromFilesStructure: false
```

#### Step 5.3: Create the dashboard JSON

```bash
touch grafana/dashboards/ml-api-dashboard.json
```

**Content** (this is a complete, working dashboard):

```json
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "green", "value": null}
            ]
          },
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
      "id": 1,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "sum(ml_predictions_total)",
          "refId": "A"
        }
      ],
      "title": "Total Predictions",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "green", "value": null},
              {"color": "yellow", "value": 0.05},
              {"color": "red", "value": 0.1}
            ]
          },
          "unit": "percentunit"
        },
        "overrides": []
      },
      "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
      "id": 2,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "sum(rate(ml_predictions_total{status=\"error\"}[5m])) / sum(rate(ml_predictions_total[5m])) or vector(0)",
          "refId": "A"
        }
      ],
      "title": "Error Rate (5m)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "green", "value": null},
              {"color": "yellow", "value": 0.5},
              {"color": "red", "value": 1}
            ]
          },
          "unit": "s"
        },
        "overrides": []
      },
      "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
      "id": 3,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le)) or vector(0)",
          "refId": "A"
        }
      ],
      "title": "P95 Latency",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "green", "value": null}
            ]
          },
          "unit": "bytes"
        },
        "overrides": []
      },
      "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
      "id": 4,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "app_memory_usage_bytes",
          "refId": "A"
        }
      ],
      "title": "Memory Usage",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {"legend": false, "tooltip": false, "viz": false},
            "lineInterpolation": "smooth",
            "lineWidth": 2,
            "pointSize": 5,
            "scaleDistribution": {"type": "linear"},
            "showPoints": "never",
            "spanNulls": false,
            "stacking": {"group": "A", "mode": "none"},
            "thresholdsStyle": {"mode": "off"}
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}]
          },
          "unit": "reqps"
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
      "id": 5,
      "options": {
        "legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom", "showLegend": true},
        "tooltip": {"mode": "multi", "sort": "desc"}
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "sum(rate(ml_predictions_total[1m])) by (model_version)",
          "legendFormat": "{{model_version}}",
          "refId": "A"
        }
      ],
      "title": "Request Rate by Model Version",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {"legend": false, "tooltip": false, "viz": false},
            "lineInterpolation": "smooth",
            "lineWidth": 2,
            "pointSize": 5,
            "scaleDistribution": {"type": "linear"},
            "showPoints": "never",
            "spanNulls": false,
            "stacking": {"group": "A", "mode": "none"},
            "thresholdsStyle": {"mode": "off"}
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}]
          },
          "unit": "s"
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
      "id": 6,
      "options": {
        "legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom", "showLegend": true},
        "tooltip": {"mode": "multi", "sort": "desc"}
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le, model_version))",
          "legendFormat": "p50 - {{model_version}}",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le, model_version))",
          "legendFormat": "p95 - {{model_version}}",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le, model_version))",
          "legendFormat": "p99 - {{model_version}}",
          "refId": "C"
        }
      ],
      "title": "Prediction Latency Percentiles",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}]
          }
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
      "id": 7,
      "options": {
        "legend": {"displayMode": "table", "placement": "right", "showLegend": true},
        "pieType": "pie",
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "tooltip": {"mode": "single", "sort": "none"}
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "sum(ml_predictions_total) by (prediction_result)",
          "legendFormat": "{{prediction_result}}",
          "refId": "A"
        }
      ],
      "title": "Predictions by Result",
      "type": "piechart"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {"legend": false, "tooltip": false, "viz": false},
            "lineInterpolation": "smooth",
            "lineWidth": 2,
            "pointSize": 5,
            "scaleDistribution": {"type": "linear"},
            "showPoints": "never",
            "spanNulls": false,
            "stacking": {"group": "A", "mode": "none"},
            "thresholdsStyle": {"mode": "off"}
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}]
          },
          "unit": "percent"
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
      "id": 8,
      "options": {
        "legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom", "showLegend": true},
        "tooltip": {"mode": "multi", "sort": "desc"}
      },
      "pluginVersion": "10.1.0",
      "targets": [
        {
          "expr": "app_cpu_usage_percent",
          "legendFormat": "CPU Usage",
          "refId": "A"
        }
      ],
      "title": "CPU Usage",
      "type": "timeseries"
    }
  ],
  "refresh": "5s",
  "schemaVersion": 38,
  "style": "dark",
  "tags": ["ml", "api", "monitoring"],
  "templating": {"list": []},
  "time": {"from": "now-15m", "to": "now"},
  "timepicker": {},
  "timezone": "browser",
  "title": "ML API Monitoring",
  "uid": "ml-api-monitoring",
  "version": 1,
  "weekStart": ""
}
```

**Explain**: "This dashboard JSON defines all our panels. Grafana loads it automatically on startup."

---

### Section 6: Running and Testing (20 minutes)

#### Step 6.1: Verify file structure

```bash
# Show the structure
find . -name "*.yml" -o -name "*.yaml" -o -name "Dockerfile" -o -name "docker-compose.yml" | head -20
```

Expected structure:
```
./Dockerfile
./docker-compose.yml
./prometheus/prometheus.yml
./prometheus/rules/alerts.yml
./grafana/provisioning/datasources/prometheus.yml
./grafana/provisioning/dashboards/dashboards.yml
./grafana/dashboards/ml-api-dashboard.json
```

#### Step 6.2: Build and start the stack

```bash
docker-compose up --build -d
```

**Show the output and explain**:
- "Docker is building our app image"
- "It's pulling the Prometheus and Grafana images"
- "All containers should start successfully"

```bash
# Check container status
docker-compose ps
```

#### Step 6.3: Access the services

In Codespaces, ports should auto-forward. Click on the PORTS tab and open:

1. **Port 5000** - ML API
   - Visit `/health` - should return `{"status": "ok"}`
   - Visit `/metrics` - should show Prometheus metrics
   - Visit `/apidocs` - Swagger UI

2. **Port 9090** - Prometheus
   - Go to Status > Targets - should show both targets as "UP"
   - Try a query: `up`
   - Try: `ml_predictions_total`

3. **Port 3000** - Grafana
   - Login with admin/admin
   - Navigate to Dashboards > ML Monitoring
   - Open "ML API Monitoring"

**Note for Codespaces**: Ports are forwarded automatically. Look for the "Ports" tab in the bottom panel.

#### Step 6.4: Generate test traffic

```bash
# Health check
curl http://localhost:5000/health

# Make predictions (use the Swagger UI or curl)
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

**Generate multiple requests**:
```bash
# Send 20 requests to generate traffic
for i in {1..20}; do
  curl -s -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"tenure":12,"MonthlyCharges":59.95,"TotalCharges":720.50,"Contract":"One year","PaymentMethod":"Electronic check","OnlineSecurity":"No","TechSupport":"No","InternetService":"DSL","gender":"Female","SeniorCitizen":"No","Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No","PaperlessBilling":"Yes","OnlineBackup":"Yes","DeviceProtection":"No","StreamingTV":"No","StreamingMovies":"No"}' &
done
wait
echo "Sent 20 requests"
```

#### Step 6.5: Explore Prometheus queries

Navigate to Prometheus (port 9090) and demonstrate:

```promql
# Basic queries
up
ml_predictions_total

# Rate calculations
rate(ml_predictions_total[1m])

# Sum by labels
sum by (model_version) (ml_predictions_total)
sum by (prediction_result) (ml_predictions_total)

# Latency percentiles
histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le))

# Average latency
rate(ml_prediction_duration_seconds_sum[5m]) / rate(ml_prediction_duration_seconds_count[5m])
```

#### Step 6.6: Show Grafana dashboard

1. Refresh the Grafana dashboard
2. Point out:
   - Total predictions counter increasing
   - Request rate graph showing traffic
   - Latency percentiles
   - Prediction result distribution (pie chart)

---

### Section 7: Wrap-up (5 minutes)

#### Key Takeaways

1. **Prometheus collects metrics** via a pull model
2. **Flask-exporter makes instrumentation easy**
3. **Grafana visualizes the data** with dashboards
4. **Alerting** can proactively notify you of issues
5. **Docker Compose** orchestrates the entire stack

#### Assignment Reminder

- Students must implement this in their own projects
- Adapt metrics to their specific ML use case
- Submit screenshots of working dashboards
- Write a brief report on their monitoring strategy

---

## Troubleshooting Guide

### Issue: "Connection refused" when accessing services

**In Codespaces**:
- Check the PORTS tab - ensure ports are forwarded
- Click "Open in Browser" for each port
- If a port shows as private, click and change to public

### Issue: Prometheus shows target as "DOWN"

```bash
# Check if app is running
docker-compose ps

# Check app logs
docker-compose logs app

# Verify metrics endpoint inside container
docker-compose exec app curl http://localhost:5000/metrics
```

### Issue: Grafana dashboard shows "No data"

1. Verify Prometheus data source is connected (Settings > Data Sources)
2. Check that Prometheus has data (query `up` in Prometheus UI)
3. Make sure some predictions have been made
4. Wait a minute for data to propagate

### Issue: Docker build fails

```bash
# Check for syntax errors
docker-compose config

# Rebuild without cache
docker-compose build --no-cache

# Check disk space
df -h
```

### Issue: Models not loading in container

```bash
# Check if models exist
ls -la models/

# Check DVC status
dvc status

# Pull models if needed
dvc pull

# Rebuild container
docker-compose up --build -d
```

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

## Files Created in This Lab

```
./Dockerfile
./docker-compose.yml
./.dockerignore
./prometheus/
  └── prometheus.yml
  └── rules/
      └── alerts.yml
./grafana/
  └── provisioning/
      └── datasources/
          └── prometheus.yml
      └── dashboards/
          └── dashboards.yml
  └── dashboards/
      └── ml-api-dashboard.json
```

**Modified Files**:
- `requirements.txt` (added monitoring dependencies)
- `src/app.py` (added Prometheus instrumentation)
