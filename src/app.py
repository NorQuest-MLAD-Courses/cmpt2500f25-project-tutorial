import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    logger.info("✅ Models and pipelines loaded successfully.")

except FileNotFoundError as e:
    logger.error(f"❌ Error loading models: {e}")
    logger.error("Please run 'dvc pull' and ensure models are in the 'models/' directory.")
    # In a real app, you might exit or return an error state
    pipeline, label_encoder, model_v1, model_v2 = None, None, None, None
except Exception as e:
    logger.error(f"❌ An unexpected error occurred: {e}")
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
            "note": "SeniorCitizen accepts both integer (0/1) and string ('No'/'Yes') formats",
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
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "PaperlessBilling": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No"
            },
            "example_alternative_senior_citizen": {
                "note": "SeniorCitizen can also be 'Yes' or 'No' (string)",
                "SeniorCitizen": "No"
            }
        }
    })

def normalize_senior_citizen(data):
    """
    Normalize SeniorCitizen to the format expected by the preprocessing pipeline.

    The pipeline expects SeniorCitizen as 0 or 1 (integer), but for user convenience,
    we accept both formats:
    - Integer: 0 or 1
    - String: "No" or "Yes" (case-insensitive)

    Args:
        data (dict): Customer data dictionary

    Returns:
        dict: Data with SeniorCitizen normalized to 0 or 1

    Raises:
        ValueError: If SeniorCitizen has an invalid value
    """
    if 'SeniorCitizen' in data:
        value = data['SeniorCitizen']

        # If it's already an integer, validate it's 0 or 1
        if isinstance(value, int):
            if value not in [0, 1]:
                raise ValueError(f"SeniorCitizen must be 0 or 1, got {value}")
            # Already correct format
            return data

        # If it's a string, convert "No"/"Yes" to 0/1
        elif isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower == "no":
                data['SeniorCitizen'] = 0
            elif value_lower == "yes":
                data['SeniorCitizen'] = 1
            else:
                raise ValueError(f"SeniorCitizen must be 'Yes', 'No', 0, or 1, got '{value}'")
            return data

        else:
            raise ValueError(f"SeniorCitizen must be string or int, got {type(value).__name__}")

    return data


def validate_input(data):
    """
    Validates the input data to ensure it has all required features
    and correct data types.

    Note: SeniorCitizen accepts both integer (0/1) and string ("No"/"Yes") formats
    for user convenience, and is automatically normalized to integer format.

    Args:
        data (dict): Customer data dictionary

    Returns:
        tuple: (error_message, status_code) - (None, 200) if valid
    """
    # Check for missing features
    missing_features = [f for f in REQUIRED_FEATURES if f not in data]
    if missing_features:
        error_msg = f"Missing required features: {', '.join(missing_features)}"
        logger.warning(f"Validation failed: {error_msg}")
        return error_msg, 400

    # Normalize SeniorCitizen before validation (accepts both 0/1 and "No"/"Yes")
    try:
        data = normalize_senior_citizen(data)
    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"Validation failed: {error_msg}")
        return error_msg, 400

    # Check data types
    for feature in NUMERICAL_FEATURES:
        if not isinstance(data[feature], (int, float)):
            # Special case for TotalCharges which might be None/null on input
            if feature == 'TotalCharges' and data[feature] is None:
                continue
            error_msg = f"Invalid type for {feature}: expected int or float, got {type(data[feature]).__name__}"
            logger.warning(f"Validation failed: {error_msg}")
            return error_msg, 400

    # Check categorical features (excluding SeniorCitizen which was already normalized)
    for feature in CATEGORICAL_FEATURES:
        if feature == 'SeniorCitizen':
            # SeniorCitizen is now normalized to int (0 or 1)
            if not isinstance(data[feature], int):
                error_msg = f"Invalid type for {feature}: expected int after normalization, got {type(data[feature]).__name__}"
                logger.warning(f"Validation failed: {error_msg}")
                return error_msg, 400
        else:
            if not isinstance(data[feature], str):
                error_msg = f"Invalid type for {feature}: expected str, got {type(data[feature]).__name__}"
                logger.warning(f"Validation failed: {error_msg}")
                return error_msg, 400

    return None, 200 # No error


def make_prediction(json_data, model, model_version):
    """
    Shared prediction logic for both v1 and v2 endpoints.

    Args:
        json_data: Input JSON data (single object or list)
        model: The trained model to use for prediction
        model_version (str): Version identifier ("v1" or "v2")

    Returns:
        tuple: (response_dict, status_code)
    """
    if json_data is None:
        logger.warning(f"{model_version}: No input data provided")
        return {"error": "No input data provided"}, 400

    # Handle empty list - return empty list
    if isinstance(json_data, list) and len(json_data) == 0:
        logger.info(f"{model_version}: Empty list provided, returning empty list")
        return [], 200

    # Check if the loaded objects are valid
    if not all([pipeline, label_encoder, model_v1, model_v2]):
        logger.error(f"{model_version}: Models or pipelines not loaded")
        return {"error": "Models or pipelines are not loaded. Check server logs."}, 500

    is_batch = isinstance(json_data, list)
    data_list = json_data if is_batch else [json_data]

    logger.info(f"{model_version}: Received {'batch' if is_batch else 'single'} prediction request with {len(data_list)} item(s)")

    # Validate all items before processing
    for item in data_list:
        # Handle potential None for TotalCharges before validation
        if 'TotalCharges' not in item or item['TotalCharges'] is None:
            item['TotalCharges'] = 0.0 # Impute with 0, pipeline will handle

        error_msg, status_code = validate_input(item)
        if error_msg:
            return {"error": error_msg}, status_code

    # If validation passes for all, proceed with prediction
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame(data_list)

        # Ensure proper dtypes to avoid pandas inference issues
        # SeniorCitizen must be int (it's categorical but stored as 0/1)
        if 'SeniorCitizen' in input_df.columns:
            input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)

        # Ensure numerical features are numeric types
        for col in NUMERICAL_FEATURES:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

        # Ensure categorical features (except SeniorCitizen) are strings
        for col in CATEGORICAL_FEATURES:
            if col in input_df.columns and col != 'SeniorCitizen':
                input_df[col] = input_df[col].astype(str)

        # Reorder columns to match pipeline's training order
        input_df = input_df[REQUIRED_FEATURES]

        # Preprocess the data
        processed_input = pipeline.transform(input_df)

        # Make prediction
        prediction_numeric = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

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
                "model_version": model_version
            })

        logger.info(f"{model_version}: Successfully generated {len(results)} prediction(s)")

        # Return single object if input was single, else return list
        return (results[0] if not is_batch else results), 200

    except Exception as e:
        # Catch-all for other errors (e.g., preprocessing issues)
        error_msg = f"An error occurred during prediction: {str(e)}"
        logger.error(f"{model_version}: {error_msg}")
        return {"error": error_msg}, 500


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
    json_data = request.get_json(silent=True)
    response_data, status_code = make_prediction(json_data, model_v1, "v1")
    return jsonify(response_data), status_code


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
    json_data = request.get_json(silent=True)
    response_data, status_code = make_prediction(json_data, model_v2, "v2")
    return jsonify(response_data), status_code


# --- 3. Run the App ---
if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug=True for development, which auto-reloads the server on code changes
    # Set host='0.0.0.0' to make the server accessible from outside the container
    app.run(host='0.0.0.0', port=port, debug=True)
