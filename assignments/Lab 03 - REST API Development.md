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

- **Problem**: `Address already in use` or `Port 5000 is in use`.
  - **Solution**: This is the most common problem in web development. It means another program is already "listening" on port 5000.
  - **Likely Causes**:
        1.  An old, "zombie" version of your own app that didn't stop properly.
        2.  An **MLflow UI** server (`mlflow ui`), which also defaults to port 5000.
        3.  On **macOS**, the "AirPlay Receiver" system service (as you saw).
        4.  On **Windows**, other system services like "Shared PnP-X IP Bus".

  - **How to Fix**: You have two options: (A) Stop the conflicting program, or (B) run your app on a different port.

  - **Fix A: Stop the Conflicting Program (Recommended)**
        You must find the Process ID (PID) of the program using the port and "kill" it.

    - **On macOS or Linux**:
            1.  Find the PID: `lsof -i :5000`
            2.  Look at the `COMMAND` and `PID` columns. If it's a `Python` process or `mlflow`, you can stop it.
            3.  Stop the process: `kill -9 [PID_NUMBER]` (e.g., `kill -9 12345`)
            4.  **macOS Specific**: If the command is `ControlCe` (Control Center), **do not** kill it. Instead, go to **System Preferences** -> **General** -> **AirDrop & Handoff** and turn **off** "AirPlay Receiver".

    - **On Windows (in Command Prompt or PowerShell)**:
            1.  Find the PID: `netstat -aon | findstr ":5000"`
            2.  Look at the last column; this is the PID.
            3.  Stop the process: `taskkill /F /PID [PID_NUMBER]` (e.g., `taskkill /F /PID 12345`)

  - **Fix B: Run Your App on a Different Port**
        The code in `src/app.py` is set up to use the `PORT` environment variable. You can just tell it to use a different port, like 5001.

    - **On macOS or Linux**:
            `PORT=5001 python src/app.py`

    - **On Windows (in PowerShell)**:
            `$env:PORT=5001; python src/app.py`

    - If you do this, remember to test with the new port!
            `curl http://127.0.0.1:5001/health`

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

---

## Task 2: Load Models and Implement the "Home" Endpoint

Now that we have a running server, our next step is to load the ML assets we created in Lab 02. We will also implement the `/{your_project}_home` endpoint, which is required by the lab to provide a human-readable "manual" for our API.

### 2.1: How and Why to Pre-Load Models

A common mistake is to load a model file from disk *inside* the prediction endpoint. This is extremely slow and inefficient.

```python
@app.route('/predict_slow')
def predict_slow():
    # BAD: Don't do this!
    model = joblib.load("model.pkl") 
    # ... predict ...
```

This code would re-load the heavyweight model file *every single time* a prediction is requested, adding seconds of delay.

The correct approach is to **load the models into memory once** when the Flask server starts. We will store them in global variables, where all our endpoints can access them instantly.

### 2.2: Update `src/app.py` to Load Assets

Let's update `src/app.py` to import `joblib` and `pandas`. We will then load our two best models (which you should have saved as `model_v1.pkl` and `model_v2.pkl` in the `models/` folder) and our preprocessing pipeline and label encoder.

```python
# Add new imports at the top
import joblib
import pandas as pd

# ... (existing Flask, jsonify, Swagger, os imports) ...

# Initialize Flask app
app = Flask(__name__)
# ... (existing Swagger config) ...

# --- Load Models and Encoders ---
# Load them ONCE when the app starts

try:
    # Load the preprocessing pipeline
    pipeline = joblib.load('data/processed/preprocessing_pipeline.pkl')
    
    # Load the label encoder
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    
    # Load the two best models
    model_v1 = joblib.load('models/model_v1.pkl')
    model_v2 = joblib.load('models/model_v2.pkl')
    
    print("✅ Models and pipelines loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading models or pipelines: {e}")
    print("Please check file paths and ensure models are saved in 'models/'")
    # In a real app, you might want to exit or use a default
    pipeline = None
    label_encoder = None
    model_v1 = None
    model_v2 = None

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
# ... (existing /health endpoint code - no changes) ...
```

### 2.3: Implement the Home Endpoint

Now we'll add the `/{your_project}_home` endpoint. This endpoint doesn't take any input; it just returns a JSON object containing documentation that explains how to use the API.

**Important**: You must replace `cmpt2500f25_tutorial_home` in the `@app.route` with your own project name, as required by the lab instructions.

Add the following code to `src/app.py`:

```python
@app.route('/cmpt2500f25_tutorial_home', methods=['GET'])
def home():
    """
    Home Endpoint
    Provides documentation and expected JSON format.
    ---
    responses:
      200:
        description: API documentation.
        schema:
          id: home_page
          properties:
            message:
              type: string
              example: "Welcome to the Telecom Churn Prediction API!"
            endpoints:
              type: object
              properties:
                health:
                  type: string
                  example: "/health"
                predict_v1:
                  type: string
                  example: "/v1/predict"
                predict_v2:
                  type: string
                  example: "/v2/predict"
            required_input_format:
              type: object
              properties:
                tenure: 
                  type: "integer"
                  example: 12
                MonthlyCharges: 
                  type: "float"
                  example: 59.99
                TotalCharges: 
                  type: "float"
                  example: 720.50
                Contract: 
                  type: "string"
                  example: "One year"
                # ... (add all other features required by your model)
                PaymentMethod:
                  type: "string"
                  example: "Electronic check"
    """
    # Define the expected JSON format (this should match your model's features)
    # This is just an example, update it with your actual features!
    example_input = {
        "tenure": 12,
        "MonthlyCharges": 59.99,
        "TotalCharges": 720.50,
        "Contract": "One year",
        "PaymentMethod": "Electronic check",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "InternetService": "DSL",
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No" 
        # ... and so on for all features
    }

    return jsonify({
        "message": "Welcome to the Telecom Churn Prediction API!",
        "api_documentation": "Use /apidocs for interactive Swagger UI.",
        "endpoints": {
            "health_check": "/health",
            "predict_v1 (Best Model)": "/v1/predict",
            "predict_v2 (2nd Best Model)": "/v2/predict"
        },
        "required_input_format": example_input
    })

# ... (existing if __name__ == '__main__' block - no changes) ...
```

### 2.3.1: 🛑 IMPORTANT: Setting Up DVC Credentials

Before you can run `dvc pull`, you may get an `Unable to locate credentials` error. This is because DVC does not have the "password" to access your DagsHub remote storage.

This is a **one-time setup** you must do on any new environment (your local machine, a new CodeSpace instance, etc.).

#### A Quick Note: `dagshub login` vs. S3 Keys

You might be tempted to run the `dagshub login` command. This command is great, but it configures credentials for Git and the DagsHub API using a system called OAuth.

Our DVC remote, however, is configured to use the **S3 protocol** (as defined in `.dvc/config`). This protocol requires a different kind of "password": an **Access Key** and a **Secret Key**.

Therefore, `dagshub login` **will not fix** the DVC `Unable to locate credentials` error for this project. You *must* use the `dvc remote modify` commands shown below to set the S3 keys.

**1. Find Your Credentials on DagsHub**:

- Go to your DagsHub repository in your browser.
- Go to **Get started with Data** section towards the bottom of the page.
- Click on **Simple data upload** tab under **Configure your data storage**.
- On the bottom right hand, you will see **Connection credentials** box.
- Find your public key ID / secret access key (they have the same value) listed as **Public Key ID and Secret Access Key** (it may be hidden, you may have to click the eye icon to make it visible; there is also a copy button next to that that will copy that to your clipboard, so you can paste in your CLI command)

- **Alternatively**, you may see a blue **"Remote"** button on your DagsHub repository page. Click it and a window will pop up. You will see your **Access Key ID** and **Secret Access Key** which you can copy.

**2. Set Your Credentials in the Terminal**:

In your terminal, run these two commands. Replace the placeholders with the keys you just copied.

```sh
dvc remote modify origin --local access_key_id <YOUR_DVC_ACCESS_KEY_ID>
dvc remote modify origin --local secret_access_key <YOUR_DVC_SECRET_ACCESS_KEY>
```

### 2.3.2: ⚠️ A Note on DVC and Model Files

Now that your credentials are set, you still must pull your data and verify your models.

**1. Pulling DVC Data**:

The code tries to load files from `data/processed/`, but this directory is tracked by DVC. If you only see a `data/processed.dvc` file, it means your data is not "pulled" from the remote.

Before running the app, you must pull your processed data:

```sh
dvc pull data/processed.dvc
```

This will download your `preprocessing_pipeline.pkl` and `label_encoder.pkl` files from your remote storage.

If you want to pull the entire `data/` directory, you can simply do:

```sh
dvc pull
```

**2. Verifying Your Models**:

The lab instructions require you to load your top two models from the `models/` folder. This folder is **not** tracked by DVC.

You must ensure that you have **manually** saved your best models from production model generation into this folder. Check that these two files exist:

- `models/model_v1.pkl` (Your best model)
- `models/model_v2.pkl` (Your second-best model)

If they do not exist, you must get them (e.g., from your `mlruns` artifacts) and save them in that location before proceeding.

**Important**: We are assuming you have followed the guide on how to generate production models guide before attempting this lab. If not, do that first and then continue here.

### 2.4: Test the New Endpoint

1. **Stop** your running server (if it's still running) with `Ctrl+C`.

2. **Ensure you have models** in the right locations.
    - `data/processed/preprocessing_pipeline.pkl`
    - `data/processed/label_encoder.pkl`
    - `models/model_v1.pkl` (Your best model from Lab 2)
    - `models/model_v2.pkl` (Your second-best model from Lab 2)
    - If your filenames are different, update the `joblib.load()` calls in the code.
  
3. **Run the application** again:

    ```sh
    python src/app.py
    ```

4. When the server starts, you should see our new print statement: `✅ Models and pipelines loaded successfully.`

5. **Test the new endpoint**:
    Open your second terminal and use `curl` (remember to change `cmpt2500f25_tutorial_home` to whatever you named your endpoint):

    ```sh
    curl http://127.0.0.1:5000/cmpt2500f25_tutorial_home
    ```

6. **Expected Output**:
    You should get a large JSON response back that includes the `message`, `endpoints`, and `required_input_format` keys.

---
