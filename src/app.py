from flask import Flask, jsonify, request
from flasgger import Swagger
import os
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Configure Flasgger for API documentation
swagger = Swagger(app)

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


if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)
