# Lab Assignment 03: REST API Development with Flask

## Overview

In this lab, we will transition from a project that runs locally to a **web service**. The goal is to create a REST API that exposes our trained machine learning models to the internet. Any user or application will be able to send new data to our API and get a churn prediction back.

We will use **Flask**, a lightweight and popular Python web framework, to build our API. We will also implement API versioning (v1 and v2) to serve different models and add automatic documentation.

### Learning Objectives

- Understand the role of a REST API in a Machine Learning system.
- Set up a **Flask** application.
- Add **API-specific dependencies** (`flask`, `flasgger`) to `requirements.txt`.
- Create a "health check" endpoint to confirm the API is running.
- Load pickled models and preprocessing pipelines into an API.
- Implement prediction endpoints that receive JSON data and return JSON predictions.
- Implement two different model versions (`/v1`, `/v2`).
- Create both manual and automatic API documentation.
- **(New)** Test a running web service from within GitHub Codespaces.

---

## What is an API? And Why Flask?

Before we write any code, let's understand *what* we are building and *why*.

### The "Why": From `predict.py` to a Web Service

Right now, your project is run using CLI commands like `python src/predict.py ...`. This is powerful, but it has a major limitation: it only works on *your* machine, for *you*.

The goal of MLOps is to make our models **useful to others**. What if a web developer wants to use your model in a new app? What if the marketing team wants to plug it into a dashboard? They can't run your Python scripts.

This is where an **API (Application Programming Interface)** comes in. An API is a "public-facing" part of your code that other applications can "talk to" over the internet.

Think of it like a **restaurant waiter** 🧑‍🍳:

- A **customer** (another app) gives the waiter an **order** (a JSON request with new data).
- The **waiter** (our API) takes the order to the **kitchen** (our ML model).
- The **kitchen** (model) prepares the **food** (a prediction).
- The **waiter** (API) brings the **food** (the JSON prediction) back to the customer.

### The "How": Flask

**Flask** is the tool we will use to build this "waiter." It is a *micro-framework* for Python, meaning it's lightweight, simple, and excellent for building web services. It will handle all the complex parts of web servers (like handling HTTP requests, routing URLs, and sending responses) so we can focus on the important part: our model's logic.

---

## Task 1: Setup API Environment and Create App

Our first task is to set up the new dependencies and create the skeleton of our Flask application. We will add `flask` (the web framework) and `flasgger` (for automatic documentation).

### 1.1: What is `flasgger`?

We are adding two tools: `flask` and `flasgger`.

- **`flask`**: This is our core API framework (the "waiter").
- **`flasgger`**: This is an amazing tool that automatically generates a beautiful, interactive documentation website for our API. It reads the docstrings in our code and builds a "Swagger UI" page. This is like giving our "waiter" an automatically-generated menu that explains every dish, what's in it, and how to order it.

### 1.2: Update `requirements.txt`

Let's add these new dependencies to `requirements.txt`.

```sh
# Add these lines under a new comment, e.g., "# API"
flask==3.0.3
flasgger==0.9.7.1
```

After adding them, make sure your virtual environment is active (`source .venv/bin/activate`) and update your installed packages:

```sh
pip install -r requirements.txt
```

### 1.3: Create the API file `src/app.py`

Now, create the main file for our API at **`src/app.py`**. We will start with just enough code to run the server and provide a `/health` endpoint.

**Why `/health`?**
This endpoint is a universal convention. Its only job is to respond `{"status": "ok"}` if the server is running. This isn't for humans; it's for *automated systems*. In production, services like Kubernetes or load balancers will "ping" this endpoint every few seconds. If they get a `200 OK` response, they know your API is "healthy" and can safely send it traffic. If the endpoint fails to respond, the system will assume your app has crashed and will automatically restart it. It's your API's "pulse check" ❤️.

```python
from flask import Flask, jsonify, request
from flasgger import Swagger
import os

# Initialize Flask app
app = Flask(__name__)

# Configure Flasgger for API documentation
swagger = Swagger(app)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health Check Endpoint
    ---
    responses:
      200:
        description: API is alive and running.
        schema:
          id: health_status
          properties:
            status:
              type: string
              example: "ok"
    """
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)
```

### 1.4: Test the Server (in GitHub Codespaces)

Running a web server inside Codespaces is a bit different from your local machine. You need to manage two terminals and "forward" your ports.

#### 1. Running the Server

1. In your VS Code terminal (where you've been running `pip`, etc.), run the app:

```sh
python src/app.py
```

1. You should see output that the server is running on port 5000.
2. **IMPORTANT**: GitHub Codespaces will detect this! A notification (a "toast") will appear in the bottom-right corner.

    - It will say "Your application running on port 5000 is available."
    - Click the **"Make Public"** button. This makes your Codespace's port 5000 accessible to the public internet (and to you).

#### 2. Testing the Server (Two Methods)

We need to test the server from a *second* terminal while the first one is busy running it.

**Method A**: Internal Test (Using `curl` in a new terminal)

1. Create a second terminal. In VS Code, you can click the "Split Terminal" button (it looks like `[ | ]`).
2. In this **new** terminal, make sure your virtual environment is active (`source .venv/bin/activate`).
3. Use the command-line tool `curl` to send an HTTP GET request to your server's health endpoint:

```sh
curl http://127.0.0.1:5000/health
```

4. **Expected Output**: You should see the JSON response immediately:

```output
{"status":"ok"}
```

**Method B**: External Test (Using your Browser)

1. After you clicked "Make Public" in step 1, go to the **"Ports"** tab in your VS Code terminal panel (it's usually next to "Terminal", "Debug Console", etc.).
2. You will see Port 5000 listed. The "Local Address" will be `http://localhost:5000` and the "Running" status will be green.
3. Copy the URL under the **"Public"** column. It will look something like `https://[your-codespace-name]-5000.app.github.dev`.
4. Paste this URL into your local computer's browser (e.g., Chrome, Firefox) and add `/health` to the end.
    - Example: `https://glowing-space-waffle-12345-5000.app.github.dev/health`
5. **Expected Output**: You should see the same `{"status": "ok"}` JSON displayed in your browser. This proves your API is live on the internet!

---

### 1.5: Troubleshooting Common Problems

- **Problem**: `ModuleNotFoundError: No module named 'flask'`
  - **Solution**: You forgot to install the requirements or activate your virtual environment.
    1. Activate the venv: `source .venv/bin/activate`
    2. Install packages: `pip install -r requirements.txt`

- **Problem**: `Address already in use` or `Port 5000 is already in use`.
  - **Solution**: This is very common. An old, "zombie" version of your server is still holding the port.
  - **Fix**: Go to the terminal where the server *was* running and press `Ctrl+C` (or `Cmd+C`) a few times. If that fails, you can use the command `lsof -i :5000` to find the Process ID (PID) and then `kill -9 [PID_NUMBER]` to force-quit it. Or, the simple fix: just reload your Codespace.

- **Problem**: `curl: (7) Failed to connect to 127.0.0.1 port 5000: Connection refused`
  - **Solution**: Your server isn't running. Check your first terminal. It has either crashed (look for an error) or you never started it. Rerun `python src/app.py`.

- **Problem**: The "Make Public" pop-up disappeared and I can't test in my browser.
  - **Solution**: Go to the **"Ports"** tab in the VS Code bottom panel. Find port 5000. Right-click on it and select "Port Visibility" -> "Public". Then you can copy the Public URL.

---

### 💡 Note for Local Machine Users (Optional)

If you are running this on your **local machine** instead of Codespaces, the process is simpler.

1. Run `python src/app.py` in Terminal 1.
2. Run `curl http://127.0.0.1:5000/health` in Terminal 2.
3. Open `http://127.0.0.1:5000/health` in your local browser.

You do not need to "Make Public" or worry about port forwarding, because it's all running on your own computer.
