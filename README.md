# Fast OpenAI API Proxy
[![GitHub Tag](https://img.shields.io/github/v/tag/HanseWare/fast-openai-api-proxy?&label=Latest)](https://github.com/HanseWare/fast-openai-api-proxy/tags)
[![Docker Hub](https://img.shields.io/badge/dockerhub-images-important.svg?&logo=Docker)](https://hub.docker.com/repository/docker/hanseware/fast-openai-api-proxy)

*Simple OpenAI entrypoint for a heterogeneously hosted set of models to mimic the OpenAI API on a single URL.*

## Overview

Fast OpenAI API Proxy is a FastAPI-based application that acts as a central proxy to various AI model providers. Its primary purpose is to aggregate multiple AI model APIs (like OpenAI, Azure OpenAI, local models, etc.) under a single, unified API interface that is compatible with the OpenAI API specification.

This approach offers several key benefits:
*   **Unified API Access:** Interact with diverse AI models from different providers using a consistent API, simplifying client-side integrations.
*   **Flexibility:** Easily switch between or combine different backend models and providers without altering client code.
*   **Client Code Simplicity:** Client applications only need to target a single API endpoint, abstracting the complexity of multiple backend integrations.

## Features

*   **OpenAI API Compatibility:** Supports key OpenAI API v1 endpoints, including:
    *   `/v1/chat/completions`
    *   `/v1/completions`
    *   `/v1/embeddings`
    *   `/v1/audio/speech`
    *   `/v1/audio/transcriptions`
    *   `/v1/audio/translations`
    *   `/v1/images/generations`
    *   `/v1/images/edits`
    *   `/v1/images/variations`
    *   `/v1/moderations`
    *   `/v1/models`
    *   `/v1/models/{model}`
*   **Support for Multiple Providers:** Configure and route requests to different AI model providers (e.g., OpenAI, Azure, custom deployments) through simple JSON configuration files.
*   **Dynamic Model Loading & Routing:** Models and their routing rules are loaded dynamically from configuration files at startup.
*   **Health Checks:** Provides `/health` for general proxy health and `/health/{model}` for specific backend model health.
*   **Streaming Support:** Supports streaming for chat completions and completions endpoints.
*   **File Uploads:** Handles file uploads for audio (transcriptions, translations) and image (edits, variations) endpoints.
*   **API Key Management:** Manages API keys for different providers via environment variables, securely injecting them into backend requests.
*   **Customizable Timeouts:** Allows configuration of request and health check timeouts at both provider and individual model endpoint levels.
*   **JSON Logging:** Structured logging using `python-json-logger` for better monitoring and debugging.

## How it Works

The proxy is built using FastAPI and works as follows:

1.  **Request Reception:** Incoming API requests are received by the FastAPI application (`app/main.py`).
2.  **API Versioning:** Requests to `/v1/*` are routed to the OpenAI v1 compatible endpoints defined in `app/api_v1.py`.
3.  **Model Configuration Lookup:** The `ModelsHandler` (`app/models_handler.py`) looks up the requested model in its loaded configuration. This configuration is derived from JSON files in a specified directory (e.g., `configs/`).
4.  **Request Proxying:** The request, along with necessary headers (including the provider-specific API key), is proxied to the target model's actual API endpoint. This is handled by utility functions in `app/utils.py` (specifically `handle_request` for most requests and `handle_file_upload` for file uploads).
5.  **Response Processing:** The response from the backend model provider is processed and returned to the client, maintaining OpenAI API compatibility.

Model configurations are loaded from `.json` files located in a directory specified by the `FOAP_CONFIG_DIR` environment variable.

## Project Structure

```
.
├── README.md               # This file
├── Dockerfile              # For building the Docker image
├── requirements.txt        # Python dependencies
├── app/                    # Core application logic
│   ├── main.py             # Main FastAPI application, health checks, lifespan
│   ├── api_v1.py           # Implements the OpenAI V1 API endpoints
│   ├── models_handler.py   # Loads and manages model configurations
│   ├── utils.py            # Helper functions for request proxying, response processing
│   └── auth.py             # Placeholder for authentication logic
├── configs/                # Directory for model configuration JSON files
│   └── openAI-example.json # Example configuration for OpenAI models
└── LICENSE                 # Project license
```

## Getting Started

### Prerequisites

*   Python (e.g., 3.8+ recommended)
*   Docker (optional, for containerized deployment)
*   Access to AI model provider APIs and their respective API keys.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HanseWare/fast-openai-api-proxy.git
    cd fast-openai-api-proxy
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

#### Environment Variables

The application can be configured using the following environment variables:

*   `FOAP_CONFIG_DIR`: (Optional) Path to the directory containing model configuration files.
    *   Defaults to `/configs` when running in Docker.
    *   Defaults to `configs` (relative to the execution path) when running locally.
*   `FOAP_LOGLEVEL`: (Optional) Log level for the application (e.g., `INFO`, `DEBUG`, `WARNING`, `ERROR`). Defaults to `INFO`.
*   `BASE_URL`: (Optional) The base URL of the proxy itself. This is used for constructing certain URLs in responses (e.g., for image data). Defaults to `http://localhost:8000`.
*   `FOAP_HOST`: (Optional) Host address for Uvicorn to bind to. Defaults to `0.0.0.0`.
*   `FOAP_PORT`: (Optional) Port for Uvicorn to bind to. Defaults to `8000`.
*   **Provider API Keys:** Environment variables containing API keys for each configured provider. The names of these variables are defined in your model configuration files (e.g., `OPENAI_API_TOKEN`).

#### Model Configuration Files

*   Create JSON files in your configuration directory (e.g., `configs/my_openai_models.json`, `configs/another_provider.json`).
*   Each file defines one or more "providers" and the "models" they offer.
*   The detailed structure of these configuration files is explained in the "Model Configuration" section below. An example can be found in `configs/openAI-example.json`.

## Usage

### Running the Proxy

#### Using Docker (Recommended)

1.  **Build the Docker image:**
    ```bash
    docker build -t fast-openai-api-proxy .
    ```
2.  **Run the Docker container:**
    Make sure to replace `your_openai_api_key` with your actual API key and adjust the volume mount for your configuration files if they are not in `./configs`.
    ```bash
    docker run -d -p 8000:8000 \
        -v $(pwd)/configs:/configs \
        -e OPENAI_API_TOKEN="your_openai_api_key" \
        -e FOAP_CONFIG_DIR="/configs" \
        -e BASE_URL="http://your_proxy_domain_or_ip:8000" \
        fast-openai-api-proxy
    ```
    *   The `-v $(pwd)/configs:/configs` mounts your local `configs` directory into the container at `/configs`.
    *   `-e OPENAI_API_TOKEN="..."` sets the API key for OpenAI. Add other `-e` flags for other provider API keys as needed.
    *   `-e FOAP_CONFIG_DIR="/configs"` tells the application inside the container where to find the config files.
    *   `-e BASE_URL` should be set to the externally accessible URL of the proxy if you are using features that return URLs (like image generation).

#### Using Uvicorn (for Development)

1.  **Set necessary environment variables:**
    ```bash
    export OPENAI_API_TOKEN="your_openai_api_key" 
    # Add other provider API keys as environment variables
    export FOAP_CONFIG_DIR="./configs" # If your configs are in ./configs
    export BASE_URL="http://localhost:8000"
    # export FOAP_LOGLEVEL="DEBUG" # For more detailed logs
    ```
2.  **Run Uvicorn:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The `--reload` flag enables auto-reloading on code changes, useful for development.

### Making API Requests

Once the proxy is running, you can make requests to it as if it were the OpenAI API. Ensure the `model` parameter in your request body matches a model name defined in your configuration files.

#### Chat Completions Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your_optional_auth_token_if_implemented" \
-d '{
    "model": "gpt-4o", # This model name must be defined in your config
    "messages": [{"role": "user", "content": "Hello! What is FastAPI?"}],
    "stream": false
}'
```

#### Image Generation Example

```bash
curl -X POST http://localhost:8000/v1/images/generations \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your_optional_auth_token_if_implemented" \
-d '{
    "model": "dall-e-3", # This model name must be defined in your config
    "prompt": "A futuristic cityscape with flying cars, digital art",
    "n": 1,
    "size": "1024x1024"
}'
```
If `BASE_URL` is configured correctly, the response will contain URLs pointing to the proxy for retrieving the image data.

### Supported OpenAI API Endpoints

The proxy aims to support the following OpenAI API v1 compatible endpoints:

*   `/v1/chat/completions`
*   `/v1/completions` (Note: OpenAI is deprecating this for their models, but the proxy can still support it for other backend providers)
*   `/v1/embeddings`
*   `/v1/audio/speech`
*   `/v1/audio/transcriptions`
*   `/v1/audio/translations`
*   `/v1/images/generations`
*   `/v1/images/edits`
*   `/v1/images/variations`
*   `/v1/moderations`
*   `/v1/models` (Lists all configured models available through the proxy)
*   `/v1/models/{model}` (Provides details for a specific configured model)
*   **(Custom)** `/v1/images/data/{model}/{file_id}`: An internal endpoint used to serve image data generated by providers that don't return direct image URLs. The `BASE_URL` environment variable must be set correctly for this to function.

### Health Check Endpoints

*   `/health`: Provides a general health status of the proxy application. Returns `{"status": "ok"}` if the proxy is running.
*   `/health/{model}`: Checks the health of the backend services for a specific, configured model. This typically involves making a test request to the target provider.

## Model Configuration

Model configurations are defined in JSON files (e.g., `my_models.json`) located in the directory specified by `FOAP_CONFIG_DIR`. Each file can contain multiple provider configurations.

The basic structure is an object where each key is a "provider name" (you choose this, e.g., "openAI", "azureAI", "localLLM").

```json
{
  "providerName1": {
    // Provider-level settings
    "models": {
      // Model-level settings
    }
  },
  "providerName2": {
    // ...
  }
}
```

### Provider Level Configuration

For each provider, you can define the following:

*   `api_key_variable`: (String, Required) The name of the environment variable that holds the API key for this provider (e.g., `"OPENAI_API_TOKEN"`, `"AZURE_API_KEY"`).
*   `prefix`: (String, Optional) A prefix that will be added to all model names defined under this provider. For example, if `prefix` is `"openai-"` and a model is named `"gpt-4o"`, it will be exposed by the proxy as `"openai-gpt-4o"`.
*   `target_base_url`: (String, Optional) A default base URL for all models under this provider. This can be overridden at the individual model endpoint level.
*   `request_timeout`: (Integer, Optional) Default request timeout in seconds for API calls to models under this provider. Defaults to 60 seconds if not set at the provider or endpoint level.
*   `health_timeout`: (Integer, Optional) Default health check timeout in seconds for models under this provider. Defaults to 60 seconds if not set at the provider or endpoint level.

### Model Level Configuration (`models` object)

Inside each provider block, the `models` object defines the specific models available from that provider. The key for each entry is the model name you will use when making requests to the proxy (e.g., `"gpt-4o"`, `"my-custom-text-model"`).

*   **Model Key** (e.g., `"gpt-4o"`):
    *   `endpoints`: (Array of Objects, Required) A list of API paths that this model supports. Each object in the array defines a specific endpoint:
        *   `path`: (String, Required) The OpenAI API compatible path this model endpoint maps to (e.g., `"v1/chat/completions"`, `"v1/embeddings"`).
        *   `target_base_url`: (String, Optional) The base URL of the actual model provider's API for this specific endpoint (e.g., `"https://api.openai.com"`, `"https://your-azure-instance.openai.azure.com"`). If not provided, `target_base_url` from the provider level is used.
        *   `target_model_name`: (String, Required) The name of the model as the target provider expects it (e.g., `"gpt-4o"`, `"text-embedding-ada-002"`).
        *   `request_timeout`: (Integer, Optional) Specific request timeout in seconds for this endpoint, overriding the provider-level setting.
        *   `health_timeout`: (Integer, Optional) Specific health check timeout in seconds for this endpoint, overriding the provider-level setting.

### Example Configuration Snippet

```json
{
  "openAI": {
    "api_key_variable": "OPENAI_API_TOKEN",
    "prefix": "openai-",
    "target_base_url": "https://api.openai.com", // Default for OpenAI models
    "request_timeout": 120, // Default request timeout for OpenAI models
    "health_timeout": 15,   // Default health check timeout for OpenAI models
    "models": {
      "gpt-4o": {
        "endpoints": [
          {
            "path": "v1/chat/completions",
            // "target_base_url" is inherited from provider if not specified
            "target_model_name": "gpt-4o"
            // "request_timeout" is inherited (120s)
          },
          {
            "path": "v1/embeddings",
            "target_model_name": "text-embedding-ada-002",
            "request_timeout": 60 // Override for this specific endpoint
          }
        ]
      },
      "dall-e-3": {
        "endpoints": [
          {
            "path": "v1/images/generations",
            "target_model_name": "dall-e-3"
            // Inherits OpenAI's target_base_url and timeouts
          }
        ]
      }
    }
  },
  "azureOpenAI": {
    "api_key_variable": "AZURE_OPENAI_KEY",
    "prefix": "azure-",
    "models": {
      "gpt-35-turbo": {
        "endpoints": [
          {
            "path": "v1/chat/completions",
            "target_base_url": "https://YOUR_AZURE_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME", // Example Azure URL
            "target_model_name": "gpt-35-turbo", // This is often ignored by Azure if deployment name is in URL
            "request_timeout": 180
          }
        ]
      }
    }
  },
  "anotherProvider": {
    "api_key_variable": "ANOTHER_PROVIDER_KEY",
    "models": {
      "custom-model-chat": {
        "endpoints": [
          {
            "path": "v1/chat/completions",
            "target_base_url": "https://api.anotherprovider.com",
            "target_model_name": "super-chat-model-v2"
          }
        ]
      }
    }
  }
}
```
In this example:
*   Requests to the proxy for model `openai-gpt-4o` on path `v1/chat/completions` will be routed to `https://api.openai.com/v1/chat/completions` using the `gpt-4o` model name for OpenAI.
*   Requests for `azure-gpt-35-turbo` will go to the specified Azure endpoint.
*   The `target_model_name` is what the backend API expects. The key used in the `models` object (e.g., `gpt-4o`, `custom-model-chat`) is what you use when calling the proxy.

## Authentication

The file `app/auth.py` contains a placeholder function `can_request(authorization: Optional[str] = Header(None))`. By default, this function returns `True`, allowing all requests.

For production environments, you should customize this function to implement proper authentication and authorization logic. This might involve:
*   Validating an API key or token passed in the `Authorization` header.
*   Checking against a list of allowed tokens or clients.
*   Integrating with an external OAuth2 provider.

Currently, unless `app/auth.py` is modified, the proxy is effectively open to any client that can reach it.

## Logging

The application uses `python-json-logger` for structured JSON logging, which is beneficial for log management and analysis systems.

*   **Log Level:** The log level can be configured using the `FOAP_LOGLEVEL` environment variable. Supported values include `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. The default is `INFO`.
*   **Key Information Logged:** Logs typically include details such as the timestamp, log level, message, requested model name, target provider, target model name, API path, and whether the request was streaming.

Example log output (formatted for readability):
```json
{
  "asctime": "2023-10-27 10:00:00,000",
  "levelname": "INFO",
  "message": "Request received",
  "request_model_name": "openai-gpt-4o",
  "target_provider": "openAI",
  "target_model_name": "gpt-4o",
  "api_path": "v1/chat/completions",
  "stream": false,
  "client_host": "127.0.0.1"
}
```

## Deployment

*   **Docker (Recommended):** The recommended method for deployment is using the provided `Dockerfile`. This ensures a consistent environment.
*   **API Key Management:** Securely manage your provider API keys. For Docker, this can be done using Docker secrets or by injecting environment variables through your container orchestration platform (e.g., Kubernetes Secrets, ECS Task Definitions).
*   **Scaling:** You can scale the proxy by running multiple instances behind a load balancer. Since the proxy is stateless, this is straightforward. Ensure your load balancer supports sticky sessions if required by any backend models (though generally not needed for typical chat/completion APIs).
*   **`BASE_URL`:** Remember to set the `BASE_URL` environment variable correctly in your deployment environment so that any URLs generated by the proxy (e.g., for image data) point to the correct external address.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to:
1.  Open an issue on the GitHub repository to discuss the change.
2.  Fork the repository, create a new branch for your feature or fix.
3.  Make your changes and commit them with clear messages.
4.  Push your branch and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
