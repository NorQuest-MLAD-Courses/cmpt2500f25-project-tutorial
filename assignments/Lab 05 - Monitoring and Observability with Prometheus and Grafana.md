# Lab 05: Monitoring and Observability with Prometheus and Grafana

## Overview

In this lab, you will enhance your containerized machine learning application by integrating Prometheus and Grafana for monitoring and observability. Building on your Docker knowledge from Lab 04, you will add monitoring capabilities to track your API's performance in real-time.

Monitoring is a critical aspect of MLOps that allows you to:

* **Monitor prediction API performance** and usage patterns
* **Track request latencies** and error rates
* **Identify system bottlenecks** and resource constraints
* **Visualize key metrics** through custom dashboards
* **Configure alerts** for potential issues before they become critical

By the end of this lab, you will have a comprehensive monitoring solution that provides insights into your machine learning API service.

## Learning Objectives

After completing this lab, you will be able to:

1. Instrument a Flask application with Prometheus metrics
2. Configure Prometheus to scrape metrics from your application
3. Set up Grafana dashboards for visualizing metrics
4. Understand the basics of PromQL (Prometheus Query Language)
5. Configure and trigger alerting rules

## Prerequisites

- Completed Labs 01-04 (working containerized Flask API)
- Understanding of Docker and docker-compose
- Familiarity with YAML configuration files

---

## Part 1: Understanding Monitoring Concepts

### What is Prometheus?

Prometheus is an open-source systems monitoring and alerting toolkit that collects and stores metrics as time-series data. Key features include:

* **Pull-based architecture**: Prometheus scrapes metrics from targets at specified intervals
* **Flexible data model**: Uses a multi-dimensional data model with key-value pairs (labels)
* **Built-in query language**: PromQL for sophisticated querying and aggregation
* **Alerting**: Define rules that trigger when conditions are met

### What is Grafana?

Grafana is an open-source visualization and analytics software that allows you to:

* Create dynamic, reusable dashboards
* Visualize metrics from various data sources (including Prometheus)
* Set up alerting rules based on metrics
* Share dashboards across teams

### Why Monitor ML Systems?

Machine learning systems have unique monitoring requirements:

| Metric Category | Examples | Why It Matters |
|-----------------|----------|----------------|
| **Request Metrics** | Request count, error rate | Service health and usage |
| **Latency Metrics** | Prediction time, p95 latency | User experience and SLAs |
| **Model Metrics** | Predictions by class, confidence | Model behavior in production |
| **System Metrics** | Memory usage, CPU | Resource planning and scaling |

### Prometheus Metric Types

| Type | Use Case | Example |
|------|----------|---------|
| **Counter** | Counts that only go up | Total requests, errors |
| **Gauge** | Values that go up and down | Memory usage, active connections |
| **Histogram** | Distribution of values | Request latency |
| **Summary** | Similar to histogram, calculates quantiles | Response size |

---

## Part 2: Instrumenting Your Flask API with Prometheus

### Step 1: Add Monitoring Dependencies

Add the following packages to your `requirements.txt`:

```
prometheus-client==0.21.0
prometheus-flask-exporter==0.23.1
psutil==6.1.0
```

### Step 2: Add Prometheus Metrics to Your Flask Application

Modify your Flask API file to include Prometheus instrumentation:

**1. Add imports at the top of your file:**

```python
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import psutil
import threading
import time
```

**2. Initialize PrometheusMetrics after creating your Flask app:**

```python
app = Flask(__name__)

# Initialize Prometheus metrics - automatically exposes /metrics endpoint
metrics = PrometheusMetrics(app)

# Add application info as a metric
metrics.info('app_info', 'ML API Information', version='1.0.0', app_name='churn-prediction-api')
```

**3. Define custom metrics for your ML application:**

```python
# Prediction counter - tracks total predictions by model version, result, and status
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_version', 'prediction_result', 'status']
)

# Prediction latency histogram - tracks how long predictions take
prediction_latency = Histogram(
    'ml_prediction_duration_seconds',
    'Time spent processing prediction requests',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# System metrics - gauges for memory and CPU
memory_usage_gauge = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage_gauge = Gauge('app_cpu_usage_percent', 'CPU usage percentage')
model_loaded_gauge = Gauge('model_loaded', 'Whether models are loaded', ['model_version'])
```

**4. Add a background thread for system resource monitoring:**

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

**5. Instrument your prediction function to record metrics:**

In your `make_prediction` function, add timing and metric recording:

```python
def make_prediction(json_data, model, model_version):
    start_time = time.time()  # Start timing

    # ... your existing validation and prediction code ...

    try:
        # ... prediction logic ...

        # Record successful predictions
        for result in results:
            prediction_counter.labels(
                model_version=model_version,
                prediction_result=result['prediction'],
                status='success'
            ).inc()

        # Record latency
        duration = time.time() - start_time
        prediction_latency.labels(model_version=model_version).observe(duration)

        return results, 200

    except Exception as e:
        # Record failed predictions
        prediction_counter.labels(
            model_version=model_version,
            prediction_result='error',
            status='error'
        ).inc()
        raise
```

**6. Start the monitoring thread when the app runs:**

```python
if __name__ == '__main__':
    # Start system resource monitoring in background
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
```

---

## Part 3: Setting Up Prometheus

### Step 1: Create Prometheus Configuration Directory

```bash
mkdir -p prometheus/rules
```

### Step 2: Create Prometheus Configuration File

Create `prometheus/prometheus.yml`:

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

  - job_name: 'ml-api'
    scrape_interval: 5s
    static_configs:
      - targets: ['app:5000']
    metrics_path: /metrics
```

### Step 3: Create Alert Rules

Create `prometheus/rules/alerts.yml`:

```yaml
groups:
  - name: ml_api_alerts
    rules:
      - alert: APIDown
        expr: up{job="ml-api"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "ML API is down"
          description: "The ML API has been unreachable for over 30 seconds."

      - alert: HighErrorRate
        expr: |
          (sum(rate(ml_predictions_total{status="error"}[2m]))
          / sum(rate(ml_predictions_total[2m]))) > 0.1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate"
          description: "Error rate is above 10%"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[2m])) by (le)) > 0.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "95th percentile latency exceeds 500ms"

  # Demo alerts - easy to trigger for lab demonstration
  - name: lab_demo_alerts
    rules:
      - alert: DemoHighErrorCount
        expr: sum(ml_predictions_total{status="error"}) >= 3
        for: 10s
        labels:
          severity: demo
        annotations:
          summary: "[DEMO] Error threshold reached"
          description: "At least 3 errors have occurred"

      - alert: DemoHighRequestRate
        expr: sum(rate(ml_predictions_total[1m])) > 0.5
        for: 30s
        labels:
          severity: demo
        annotations:
          summary: "[DEMO] High request rate"
          description: "More than 0.5 requests per second"
```

---

## Part 4: Setting Up Grafana

### Step 1: Create Grafana Directory Structure

```bash
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards
```

### Step 2: Configure Prometheus Data Source

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
```

### Step 3: Configure Dashboard Provisioning

Create `grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'ML Monitoring Dashboards'
    orgId: 1
    folder: 'ML Monitoring'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
```

### Step 4: Create a Dashboard

Create `grafana/dashboards/ml-api-dashboard.json` with panels for:
- API Status (up/down)
- Total Predictions count
- Error Rate percentage
- P95 Latency
- Memory Usage
- Request Rate over time
- Latency Percentiles graph
- Predictions by Result (pie chart)
- Active Alerts list

(A complete dashboard JSON will be provided by your instructor or you can create one using the Grafana UI)

---

## Part 5: Update Docker Compose

Update your `docker-compose.yml` to include Prometheus and Grafana services:

```yaml
version: '3.8'

services:
  app:
    # ... your existing app configuration ...

  mlflow:
    # ... your existing mlflow configuration ...

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
      - '--web.enable-lifecycle'
    networks:
      - ml-network
    depends_on:
      - app
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
    networks:
      - ml-network
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  # ... existing volumes ...
  prometheus-data:
  grafana-data:
```

---

## Part 6: Running and Testing

### Step 1: Build and Start All Services

```bash
docker-compose up --build -d
docker-compose ps  # Verify all containers are running
```

### Step 2: Access the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| ML API | http://localhost:5000 | N/A |
| ML API Metrics | http://localhost:5000/metrics | N/A |
| Prometheus | http://localhost:9090 | N/A |
| Grafana | http://localhost:3000 | admin/admin |
| MLflow | http://localhost:5001 | N/A |

### Step 3: Generate Test Traffic

```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"your": "test_data"}'

# Generate multiple requests for metrics
for i in {1..20}; do
  curl -s -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"your": "test_data"}' &
done
wait
```

### Step 4: Explore Prometheus Queries

Navigate to http://localhost:9090 and try these PromQL queries:

```promql
# Current API status
up{job="ml-api"}

# Total predictions
ml_predictions_total

# Request rate per second
rate(ml_predictions_total[1m])

# Average latency
rate(ml_prediction_duration_seconds_sum[5m]) / rate(ml_prediction_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le))
```

### Step 5: Trigger Alerts for Demonstration

To trigger the demo alerts:

```bash
# Trigger DemoHighErrorCount - send requests with invalid data
for i in {1..5}; do
  curl -s -X POST http://localhost:5000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}'
done

# Check alerts in Prometheus: http://localhost:9090/alerts
```

---

## Part 7: PromQL Reference

### Common Query Patterns

| Query | Description |
|-------|-------------|
| `metric_name` | Current value |
| `rate(counter[5m])` | Per-second rate over 5 minutes |
| `sum by(label) (metric)` | Sum grouped by label |
| `histogram_quantile(0.95, ...)` | 95th percentile |
| `increase(counter[1h])` | Total increase over 1 hour |

### Aggregation Operators

- `sum` - Sum all values
- `avg` - Average
- `min` / `max` - Minimum/Maximum
- `count` - Count of elements

---

## Deliverables

1. **GitHub Repository** with:
   - Instrumented Flask application with Prometheus metrics
   - Prometheus configuration files
   - Grafana provisioning and dashboard files
   - Updated docker-compose.yml

2. **Screenshots** showing:
   - Prometheus targets page (all targets "UP")
   - PromQL query results in Prometheus
   - Grafana dashboard with metrics
   - At least one alert firing in Prometheus

3. **Brief Report** explaining:
   - What metrics you chose to track and why
   - How monitoring helps detect issues in ML systems
   - Any challenges and how you resolved them

---

## Evaluation Criteria

| Criteria | Points |
|----------|--------|
| Prometheus metrics correctly implemented | 25 |
| Prometheus configuration and alerting | 20 |
| Grafana dashboard setup | 20 |
| Successfully triggered and documented alerts | 15 |
| Report quality and understanding | 15 |
| Code quality and best practices | 5 |

---

## Troubleshooting

### Common Issues

1. **Prometheus target shows "DOWN"**
   - Verify Flask app binds to `0.0.0.0`, not `127.0.0.1`
   - Check that `/metrics` endpoint is accessible
   - Verify container names match in prometheus.yml

2. **No data in Grafana**
   - Wait for Prometheus to scrape (check scrape interval)
   - Verify data source is configured correctly
   - Test queries in Prometheus first

3. **Alerts not firing**
   - Check alert rules syntax in Prometheus UI
   - Verify the conditions are actually being met
   - Check the `for` duration hasn't elapsed

---

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [prometheus-flask-exporter](https://github.com/rycus86/prometheus_flask_exporter)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
