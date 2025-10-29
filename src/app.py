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
