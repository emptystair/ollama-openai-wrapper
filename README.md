# Ollama-OpenAI API Wrapper

This project provides a compatibility layer between Ollama and applications that use the OpenAI API. It allows you to use local Ollama models with software designed for OpenAI's API (like Cursor, ChatGPT clients, etc.).

## Features

- Translates OpenAI API calls to Ollama API calls
- Supports streaming responses
- Dynamic model selection
- Health check endpoint
- Compatible with Cursor AI and other OpenAI API clients

## Quick Start

### Prerequisites
- [Docker](https://www.docker.com/get-started/)
- [Ollama](https://ollama.ai/) installed and running
  - You'll need to have a local model pulled in Ollama (e.g., `ollama pull llama3`)
- [ngrok](https://ngrok.com/) for exposing your local server (Cursor will not recognize a localhost address)

### Running with Docker

1. Clone or download this repository

2. Build the Docker image:
   ```bash
   docker build -t openai-wrapper .
   ```

3. Run the container:
   ```bash
   docker run --network host openai-wrapper
   ```

   > Note: Using `--network host` allows the container to access Ollama running on your local machine.

4. The server will be running at `http://localhost:5000`

### Exposing to the Internet (Optional)

If you want to use this service with cloud-based tools, you can expose it using ngrok:

```bash
ngrok http 5000
```

Copy the HTTPS URL provided by ngrok (e.g., `https://85d9-2603-6080-xxxx-xxxx.ngrok-free.app`).

## Using with Cursor AI

1. Open Cursor AI settings
2. Set the following:
   - API Base: Your server URL + `/v1` (e.g., `https://85d9-2603-6080-xxxx-xxxx.ngrok-free.app/v1`)
   - API Key: Any string (e.g., `fake-api-key`)
3. Click "Add Model"
4. Enter the exact name of an Ollama model you have installed
5. Deselect all other models
6. Save and start using your local Ollama models in Cursor!

## Available Endpoints

- `/v1/chat/completions` - Chat completions endpoint compatible with OpenAI's API
- `/v1/models` - Lists available models from your Ollama installation
- `/health` - Health check endpoint

## Environment Variables

- `OLLAMA_HOST` - Hostname or IP of your Ollama server (default: `localhost`)
- `API_KEY` - API key to validate requests (not strictly enforced)

## Troubleshooting

- Ensure Ollama is running and has models downloaded (`ollama pull modelname`)
- Check the logs of the Docker container for any errors
- Make sure your firewall allows connections to port 5000
- If using ngrok, verify the tunnel is active
