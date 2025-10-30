import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger
import os

# Initialize Flask app
app = Flask(__name__)

# Configure Swagger for API documentation
# We use a template to define the basic structure
template = {
    "swagger": "2.0",
    "info": {
        "title": "CMPT 2500 F25 Project Tutorial - Churn Prediction API",
        "description": "An API to predict customer churn using machine learning models. \
            This API is part of the CMPT 2500 F25 Project Tutorial.",
        "version": "1.0.0"
    },
    "tags": [
        {
            "name": "Health Check",
            "description": "Endpoint to check if the API is running."
        },
        {
            "name": "API Information",
            "description": "Endpoint to get API usage information."
        },
        {
            "name": "Prediction Endpoints",
            "description": "Endpoints for making predictions."
        }
    ]
}
swagger = Swagger(app, template=template)


# --- 1. Load Models and Preprocessing Objects ---
# We load these objects once at startup to save time on each request
try:
    pipeline = joblib.load('data/processed/preprocessing_pipeline.pkl')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    model_v1 = joblib.load('models/model_v1.pkl')
    model_v2 = joblib.load('models/model_v2.pkl')
    
    print("✅ Models and pipelines loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("Please run 'dvc pull' and ensure models are in the 'models/' directory.")
    # In a real app, you might exit or return an error state
    pipeline, label_encoder, model_v1, model_v2 = None, None, None, None
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    pipeline, label_encoder, model_v1, model_v2 = None, None, None, None

# Define the expected features and their types
# Based on our preprocessing pipeline
REQUIRED_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges",
    "SeniorCitizen"
]

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [f for f in REQUIRED_FEATURES if f not in NUMERICAL_FEATURES]


# --- 2. Define API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health Check
    ---
    tags:
      - Health Check
    description: Responds with a 200 OK status if the API is running.
    responses:
      200:
        description: API is running
        schema:
          type: object
          properties:
            status:
              type: string
              example: "ok"
    """
    # Simple 200 OK response
    return jsonify({"status": "ok"}), 200

@app.route('/cmpt2500f25_tutorial_home', methods=['GET'])
def home():
    """
    API Information
    ---
    tags:
      - API Information
    description: Provides usage information and the required input format.
    responses:
      200:
        description: API information
        schema:
          type: object
          properties:
            message:
              type: string
            api_documentation:
              type: string
            model_versions_available:
              type: array
              items:
                type: string
            required_input_format:
              type: object
    """
    # This endpoint returns a JSON object with API information
    # It's a good place to tell users how to use your API
    return jsonify({
        "message": "Welcome to the CMPT 2500 F25 Project Tutorial API!",
        "api_documentation": "Find the interactive documentation at /apidocs/",
        "model_versions_available": ["/v1/predict", "/v2/predict"],
        "required_input_format": {
            "description": "A JSON object (or list of objects) with all 19 features.",
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "example_single_record": {
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
            }
        }
    })

def validate_input(data):
    """
    Validates the input data to ensure it has all required features
    and correct data types.
    """
    # Check for missing features
    missing_features = [f for f in REQUIRED_FEATURES if f not in data]
    if missing_features:
        return f"Missing required features: {', '.join(missing_features)}", 400

    # Check data types
    for feature in NUMERICAL_FEATURES:
        if not isinstance(data[feature], (int, float)):
            # Special case for TotalCharges which might be None/null on input
            if feature == 'TotalCharges' and data[feature] is None:
                continue
            return f"Invalid type for {feature}: expected int or float, got {type(data[feature]).__name__}", 400
            
    for feature in CATEGORICAL_FEATURES:
        if not isinstance(data[feature], str):
            # In preprocess.py, SeniorCitizen (0/1) is mapped to "No"/"Yes".
            # The pipeline is trained on the string "No"/"Yes".
            # Our API validation must therefore expect a string.
            return f"Invalid type for {feature}: expected str, got {type(data[feature]).__name__}", 400

    return None, 200 # No error


@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """
    Make a prediction using Model v1 (Best Model)
    ---
    tags:
      - Prediction Endpoints
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: >
            Customer data for churn prediction.
            Can be a single JSON object or a list of JSON objects.
        required: true
        schema:
          type: "object"
          properties:
            tenure:
              type: "integer"
              example: 12
            MonthlyCharges:
              type: "number"
              example: 59.95
            TotalCharges:
              type: "number"
              example: 720.50
            Contract:
              type: "string"
              example: "One year"
            PaymentMethod:
              type: "string"
              example: "Electronic check"
            OnlineSecurity:
              type: "string"
              example: "No"
            TechSupport:
              type: "string"
              example: "No"
            InternetService:
              type: "string"
              example: "DSL"
            gender:
              type: "string"
              example: "Female"
            SeniorCitizen:
              type: "string"
              example: "No"
            Partner:
              type: "string"
              example: "Yes"
            Dependents:
              type: "string"
              example: "No"
            PhoneService:
              type: "string"
              example: "Yes"
            MultipleLines:
              type: "string"
              example: "No"
            PaperlessBilling:
              type: "string"
              example: "Yes"
            OnlineBackup:
              type: "string"
              example: "Yes"
            DeviceProtection:
              type: "string"
              example: "No"
            StreamingTV:
              type: "string"
              example: "No"
            StreamingMovies:
              type: "string"
              example: "No"
    responses:
      200:
        description: Prediction successful
        schema:
          type: "object"
          properties:
            prediction:
              type: "string"
              example: "No"
            probability:
              type: "number"
              example: 0.85
            model_version:
              type: "string"
              example: "v1"
      400:
        description: Invalid input data
        schema:
          type: "object"
          properties:
            error:
              type: "string"
              example: "Missing required features: tenure"
      500:
        description: Internal server error
        schema:
          type: "object"
          properties:
            error:
              type: "string"
              example: "Models or pipelines are not loaded. Check server logs."
    """
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400

    # Check if the loaded objects are valid
    if not all([pipeline, label_encoder, model_v1, model_v2]):
         return jsonify({"error": "Models or pipelines are not loaded. Check server logs."}), 500

    is_batch = isinstance(json_data, list)
    data_list = json_data if is_batch else [json_data]

    # Validate all items before processing
    for item in data_list:
        # Handle potential None for TotalCharges before validation
        if 'TotalCharges' not in item or item['TotalCharges'] is None:
            item['TotalCharges'] = 0.0 # Impute with 0, pipeline will handle
            
        error_msg, status_code = validate_input(item)
        if error_msg:
            return jsonify({"error": error_msg}), status_code

    # If validation passes for all, proceed with prediction
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame(data_list)
        
        # Reorder columns to match pipeline's training order
        input_df = input_df[REQUIRED_FEATURES]

        # Preprocess the data
        processed_input = pipeline.transform(input_df)

        # Make prediction
        prediction_numeric = model_v1.predict(processed_input)
        prediction_proba = model_v1.predict_proba(processed_input)

        # Decode prediction
        prediction_label = label_encoder.inverse_transform(prediction_numeric)
        
        # Format output
        results = []
        for i in range(len(prediction_label)):
            # Get the probability of the *predicted* class
            probability = prediction_proba[i][prediction_numeric[i]]
            results.append({
                "prediction": prediction_label[i],
                "probability": float(probability),
                "model_version": "v1"
            })
        
        # Return single object if input was single, else return list
        return jsonify(results[0] if not is_batch else results)

    except Exception as e:
        # Catch-all for other errors (e.g., preprocessing issues)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """
    Make a prediction using Model v2 (2nd Best Model)
    ---
    tags:
      - Prediction Endpoints
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: >
            Customer data for churn prediction.
            Can be a single JSON object or a list of JSON objects.
        required: true
        schema:
          type: "object"
          properties:
            tenure:
              type: "integer"
              example: 12
            MonthlyCharges:
              type: "number"
              example: 59.95
            TotalCharges:
              type: "number"
              example: 720.50
            Contract:
              type: "string"
              example: "One year"
            PaymentMethod:
              type: "string"
              example: "Electronic check"
            OnlineSecurity:
              type: "string"
              example: "No"
            TechSupport:
              type: "string"
              example: "No"
            InternetService:
              type: "string"
              example: "DSL"
            gender:
              type: "string"
              example: "Female"
            SeniorCitizen:
              type: "string"
              example: "No"
            Partner:
              type: "string"
              example: "Yes"
            Dependents:
              type: "string"
              example: "No"
            PhoneService:
              type: "string"
              example: "Yes"
            MultipleLines:
              type: "string"
              example: "No"
            PaperlessBilling:
              type: "string"
              example: "Yes"
            OnlineBackup:
              type: "string"
              example: "Yes"
            DeviceProtection:
              type: "string"
              example: "No"
            StreamingTV:
              type: "string"
              example: "No"
            StreamingMovies:
              type: "string"
              example: "No"
    responses:
      200:
        description: Prediction successful
        schema:
          type: "object"
          properties:
            prediction:
              type: "string"
              example: "No"
            probability:
              type: "number"
              example: 0.85
            model_version:
              type: "string"
              example: "v2"
      400:
        description: Invalid input data
        schema:
          type: "object"
          properties:
            error:
              type: "string"
              example: "Missing required features: tenure"
      500:
        description: Internal server error
        schema:
          type: "object"
          properties:
            error:
              type: "string"
              example: "Models or pipelines are not loaded. Check server logs."
    """
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400

    # Check if the loaded objects are valid
    if not all([pipeline, label_encoder, model_v1, model_v2]):
         return jsonify({"error": "Models or pipelines are not loaded. Check server logs."}), 500

    is_batch = isinstance(json_data, list)
    data_list = json_data if is_batch else [json_data]

    # Validate all items before processing
    for item in data_list:
        # Handle potential None for TotalCharges before validation
        if 'TotalCharges' not in item or item['TotalCharges'] is None:
            item['TotalCharges'] = 0.0 # Impute with 0, pipeline will handle
            
        error_msg, status_code = validate_input(item)
        if error_msg:
            return jsonify({"error": error_msg}), status_code

    # If validation passes for all, proceed with prediction
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame(data_list)

        # Reorder columns to match pipeline's training order
        input_df = input_df[REQUIRED_FEATURES]
        
        # Preprocess the data
        processed_input = pipeline.transform(input_df)

        # Make prediction
        prediction_numeric = model_v2.predict(processed_input)
        prediction_proba = model_v2.predict_proba(processed_input)

        # Decode prediction
        prediction_label = label_encoder.inverse_transform(prediction_numeric)
        
        # Format output
        results = []
        for i in range(len(prediction_label)):
            # Get the probability of the *predicted* class
            probability = prediction_proba[i][prediction_numeric[i]]
            results.append({
                "prediction": prediction_label[i],
                "probability": float(probability),
                "model_version": "v2"
            })
        
        # Return single object if input was single, else return list
        return jsonify(results[0] if not is_batch else results)

    except Exception as e:
        # Catch-all for other errors (e.g., preprocessing issues)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


# --- 3. Run the App ---
if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug=True for development, which auto-reloads the server on code changes
    # Set host='0.0.0.0' to make the server accessible from outside the container
    app.run(host='0.0.0.0', port=port, debug=True)
