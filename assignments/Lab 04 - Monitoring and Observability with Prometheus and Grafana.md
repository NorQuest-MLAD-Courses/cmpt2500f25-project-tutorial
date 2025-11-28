# Lab 04: Monitoring and Observability with Prometheus and Grafana

## Overview

In this lab, you will enhance your containerized machine learning application by integrating Prometheus and Grafana for monitoring and observability. This is a critical aspect of MLOps that allows you to:

* **Monitor prediction API performance** and usage patterns
* **Track request latencies** and error rates
* **Identify system bottlenecks** and resource constraints
* **Visualize key metrics** through custom dashboards
* **Configure alerts** for potential issues

By the end of this lab, you will have a comprehensive monitoring solution that provides insights into your machine learning API service.

## Learning Objectives

After completing this lab, you will be able to:

1. Instrument a Flask application with Prometheus metrics
2. Configure Prometheus to scrape metrics from your application
3. Set up Grafana dashboards for visualizing metrics
4. Understand the basics of PromQL (Prometheus Query Language)
5. Configure basic alerting rules

## Prerequisites

- Completed Labs 01-03 (working Flask API)
- Basic understanding of Docker and docker-compose
- Familiarity with YAML configuration files

---

## Part 1: Understanding Monitoring Concepts (10 minutes)

### What is Prometheus?

Prometheus is an open-source systems monitoring and alerting toolkit that collects and stores metrics as time-series data. Key features include:

* **Pull-based architecture**: Prometheus scrapes metrics from targets at specified intervals
* **Flexible data model**: Uses a multi-dimensional data model with key-value pairs (labels)
* **Built-in query language**: PromQL for sophisticated querying and aggregation
* **No reliance on distributed storage**: Stores data locally

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

---

## Part 2: Instrumenting Your Flask API (25 minutes)

### Step 1: Add Monitoring Dependencies

Add the following packages to your `requirements.txt`:

```
prometheus-client==0.21.0
prometheus-flask-exporter==0.23.1
psutil==6.1.0
```

### Step 2: Add Prometheus Metrics to Your Flask Application

Modify your Flask API file to include Prometheus instrumentation. You'll need to:

1. **Import the necessary libraries**:
   ```python
   from prometheus_flask_exporter import PrometheusMetrics
   from prometheus_client import Counter, Histogram, Gauge
   import psutil
   import os
   import threading
   import time
   ```

2. **Initialize PrometheusMetrics** after creating your Flask app:
   ```python
   app = Flask(__name__)
   metrics = PrometheusMetrics(app)
   ```

3. **Define custom metrics** for your ML application:
   ```python
   # Prediction metrics
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
   memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
   cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage')
   ```

4. **Instrument your prediction endpoints** to record metrics:
   ```python
   @app.route('/v1/predict', methods=['POST'])
   def predict_v1():
       start_time = time.time()
       try:
           # Your prediction logic here
           result, status_code = make_prediction(...)

           # Record successful prediction
           prediction_counter.labels(
               model_version='v1',
               prediction_result=result.get('prediction', 'unknown'),
               status='success'
           ).inc()

           # Record latency
           duration = time.time() - start_time
           prediction_latency.labels(model_version='v1').observe(duration)

           return jsonify(result), status_code
       except Exception as e:
           prediction_counter.labels(
               model_version='v1',
               prediction_result='error',
               status='error'
           ).inc()
           raise
   ```

5. **Add a background thread for system metrics**:
   ```python
   def monitor_system_resources():
       """Background thread to monitor system resources."""
       while True:
           process = psutil.Process(os.getpid())
           memory_usage.set(process.memory_info().rss)
           cpu_usage.set(process.cpu_percent(interval=1))
           time.sleep(15)

   # Start the monitoring thread (add to app startup)
   monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
   monitor_thread.start()
   ```

### Understanding Prometheus Metric Types

| Type | Use Case | Example |
|------|----------|---------|
| **Counter** | Counts that only go up | Total requests, errors |
| **Gauge** | Values that go up and down | Memory usage, active connections |
| **Histogram** | Distribution of values | Request latency |
| **Summary** | Similar to histogram, calculates quantiles | Response size |

---

## Part 3: Containerizing Your Application (20 minutes)

### Step 1: Create a Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/

# Expose the API port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "src.app"]
```

### Step 2: Create docker-compose.yml

Create a `docker-compose.yml` file:

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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
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
      - monitoring-network

  grafana:
    image: grafana/grafana:latest
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

networks:
  monitoring-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

---

## Part 4: Configuring Prometheus (15 minutes)

### Step 1: Create Prometheus Configuration

Create the directory structure:
```bash
mkdir -p prometheus/rules
```

Create `prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  # Scrape Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Scrape your ML API
  - job_name: 'ml-api'
    scrape_interval: 5s
    static_configs:
      - targets: ['app:5000']
    metrics_path: /metrics
```

### Step 2: Create Basic Alert Rules (Optional)

Create `prometheus/rules/alerts.yml`:

```yaml
groups:
  - name: ml_api_alerts
    rules:
      - alert: HighErrorRate
        expr: sum(rate(ml_predictions_total{status="error"}[5m])) / sum(rate(ml_predictions_total[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate detected"
          description: "Error rate is above 10% for the last 5 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le)) > 1.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
          description: "95th percentile latency is above 1 second"

      - alert: APIDown
        expr: up{job="ml-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML API is down"
          description: "The ML API has been unreachable for more than 1 minute"
```

---

## Part 5: Setting Up Grafana (25 minutes)

### Step 1: Create Grafana Provisioning Structure

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
  - name: 'ML Monitoring'
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
- Request rate over time
- Error rate percentage
- Prediction latency (p50, p95, p99)
- Predictions by model version
- System resource usage

(A complete dashboard JSON template will be provided by your instructor)

---

## Part 6: Running the Monitoring Stack (15 minutes)

### Step 1: Build and Start Services

```bash
# Build and start all services
docker-compose up --build -d

# Check that all containers are running
docker-compose ps

# View logs if needed
docker-compose logs -f
```

### Step 2: Verify the Setup

1. **Check your API**: http://localhost:5000/health
2. **Check Prometheus**: http://localhost:9090
3. **Check Grafana**: http://localhost:3000 (login: admin/admin)

### Step 3: Generate Test Traffic

Send some test requests to generate metrics:

```bash
# Health check
curl http://localhost:5000/health

# Make predictions (use your actual test data format)
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"your": "test_data"}'
```

### Step 4: Explore Metrics in Prometheus

Navigate to http://localhost:9090 and try these queries:

```promql
# Total predictions
ml_predictions_total

# Request rate per second
rate(ml_predictions_total[1m])

# Average prediction latency
rate(ml_prediction_duration_seconds_sum[5m]) / rate(ml_prediction_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, sum(rate(ml_prediction_duration_seconds_bucket[5m])) by (le))
```

---

## Part 7: Understanding PromQL Basics

### Common Query Patterns

| Query | Description |
|-------|-------------|
| `metric_name` | Current value of a metric |
| `rate(counter[5m])` | Per-second rate over 5 minutes |
| `sum by(label) (metric)` | Sum grouped by label |
| `histogram_quantile(0.95, ...)` | 95th percentile |
| `increase(counter[1h])` | Total increase over 1 hour |

### Aggregation Operators

- `sum` - Sum all values
- `avg` - Average
- `min` / `max` - Minimum/Maximum
- `count` - Count of elements
- `topk(n, ...)` - Top N values

---

## Deliverables

Submit the following:

1. **GitHub Repository Link** with your updated code including:
   - Instrumented Flask application with Prometheus metrics
   - Dockerfile and docker-compose.yml
   - Prometheus configuration files
   - Grafana provisioning and dashboard files

2. **Screenshots** showing:
   - Prometheus targets page (all targets "UP")
   - At least one PromQL query result in Prometheus
   - Your Grafana dashboard with visible metrics

3. **Brief Report** (1-2 pages) explaining:
   - What metrics you chose to track and why
   - How monitoring helps detect issues in ML systems
   - Any challenges you encountered and how you resolved them

---

## Evaluation Criteria

| Criteria | Points |
|----------|--------|
| Prometheus metrics correctly implemented in Flask app | 25 |
| Docker configuration (Dockerfile + docker-compose) | 20 |
| Prometheus configuration and scraping | 15 |
| Grafana dashboard setup and visualization | 20 |
| Report quality and understanding of concepts | 15 |
| Code quality and best practices | 5 |

---

## Troubleshooting

### Common Issues

1. **"Connection refused" when Prometheus scrapes the app**
   - Ensure your Flask app binds to `0.0.0.0`, not `127.0.0.1`
   - Check that the port in docker-compose matches your app

2. **Metrics endpoint returns 404**
   - Verify `PrometheusMetrics(app)` is initialized
   - Check that `/metrics` endpoint exists

3. **Grafana can't connect to Prometheus**
   - Use `http://prometheus:9090` (container name), not `localhost`
   - Ensure both are on the same Docker network

4. **No data in Grafana dashboard**
   - Wait for Prometheus to scrape (check scrape interval)
   - Verify the data source is configured correctly
   - Test queries directly in Prometheus first

---

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [prometheus-flask-exporter Documentation](https://github.com/rycus86/prometheus_flask_exporter)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
