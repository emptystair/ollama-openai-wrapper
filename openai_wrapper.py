import os
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import time
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Authorization", "Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

# Change to use localhost instead of hardcoded IP
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:11434/api/generate"


# Define fetch_available_models BEFORE using it
def fetch_available_models():
    """Fetch available models from Ollama."""
    try:
        # Change to use localhost instead of hardcoded IP
        response = requests.get(f"http://{OLLAMA_HOST}:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            logger.error(f"Failed to fetch models: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return []

# Now MODEL_NAME can safely use fetch_available_models
MODEL_NAME = fetch_available_models()[0] if fetch_available_models() else "deepseek-r1:32b"

def parse_ollama_response(response_text):
    """Parse the Ollama response text and handle any potential errors."""
    try:
        response_data = json.loads(response_text)
        response_text = response_data.get("response", "")
        return response_text, response_data.get("done", True)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response: {e}")
        return None, False
    except Exception as e:
        logger.error(f"Error processing Ollama response: {e}")
        return None, False

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completion requests."""
    try:
        # Check Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                "error": {
                    "message": "Invalid API key provided.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }), 401

        # Extract request data
        data = request.json
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        model = data.get("model", MODEL_NAME)  # Allow model selection

        # Construct prompt from messages
        prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
        
        # Send request to Ollama
        ollama_request = {
            "model": model,  # Use the selected model
            "prompt": prompt,
            "stream": stream
        }

        logger.debug(f"Sending request to Ollama: {json.dumps(ollama_request, indent=2)}")
        ollama_response = requests.post(
            OLLAMA_API_URL,
            json=ollama_request,
            stream=stream,
            timeout=30
        )

        if stream:
            def generate():
                for chunk in ollama_response.iter_lines():
                    if chunk:
                        # Parse the chunk and format it like OpenAI's streaming format
                        parsed_chunk = json.loads(chunk)
                        response_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:6]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,  # Use the selected model, not MODEL_NAME
                            "choices": [{
                                "delta": {
                                    "content": parsed_chunk.get("response", ""),
                                },
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                
                # Send the final chunk
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:6]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,  # Use the selected model, not MODEL_NAME
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate(), mimetype='text/event-stream')

        # Non-streaming response handling (your existing code)
        logger.debug(f"Ollama Response: {ollama_response.text}")

        response_text, done = parse_ollama_response(ollama_response.text)
        if response_text is None:
            return jsonify({
                "error": {
                    "message": "Failed to parse Ollama response",
                    "type": "internal_error"
                }
            }), 500

        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:6]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,  # Use the selected model, not MODEL_NAME
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(prompt) + len(response_text)) // 4,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"\n\n{response_text}"
                    },
                    "logprobs": None,
                    "finish_reason": "stop" if done else "length",
                    "index": 0
                }
            ]
        }

        logger.debug(f"Sending response: {json.dumps(response_data, indent=2)}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }), 500

def is_valid_api_key(api_key):
    """Validate the API key."""
    # Replace this with your actual API key validation
    # You should store valid keys in environment variables or a secure database
    valid_keys = [os.getenv('API_KEY', 'your-default-api-key')]
    return api_key in valid_keys

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Present compatible model information to Cursor."""
    models = fetch_available_models()
    if not models:
        return jsonify({
            "error": {
                "message": "Failed to fetch models from Ollama",
                "type": "internal_error"
            }
        }), 500

    model_data = [
        {
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai",
            "permission": [],
            "root": model,
            "parent": None
        }
        for model in models
    ]
    return jsonify({
        "object": "list",
        "data": model_data
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status."""
    try:
        # Change to use localhost instead of hardcoded IP
        ollama_response = requests.get(f"http://{OLLAMA_HOST}:11434/api/tags")
        if ollama_response.status_code == 200:
            return jsonify({
                "status": "healthy",
                "ollama": "connected",
                "models": ollama_response.json()
            }), 200
        else:
            return jsonify({
                "status": "degraded",
                "ollama": "error",
                "error": f"Ollama returned status code {ollama_response.status_code}"
            }), 200
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "degraded",
            "ollama": "disconnected",
            "error": str(e)
        }), 200

if __name__ == '__main__':
    logger.info("Starting server...")
    print(f"Before using this server, make sure to pull the Ollama model using:")
    print(f"ollama pull ")
    app.run(host='0.0.0.0', port=5000, debug=True)