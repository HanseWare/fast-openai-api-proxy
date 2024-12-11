import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import os
from typing import Dict, Optional, Any, List, Set
import logging
from pythonjsonlogger.json import JsonFormatter

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response

from auth import can_request
from models_app import handler as models
__name__ = "hanseware.fast-openai-api-proxy"

RESERVED_ATTRS: List[str] = [
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
]

logger = logging.getLogger(__name__)


class FOAP(FastAPI):
    base_url: str
    model_handler: Any
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_handler = models
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000")


def setup_logging():
    loglevel = os.getenv("FOAP_LOGLEVEL", "INFO").upper()
    logger.info("Setting log level from env to", loglevel)
    logging.basicConfig(level=logging.getLevelName(loglevel))
    logHandler = logging.StreamHandler()
    formatter = JsonFormatter(timestamp=True, reserved_attrs=RESERVED_ATTRS, datefmt='%Y-%m-%d %H:%M:%S')
    logHandler.setFormatter(formatter)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logHandler)
    uvi_logger = logging.getLogger("uvicorn.access")
    uvi_logger.handlers.clear()
    uvi_logger.addHandler(logHandler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FOAP(lifespan=lifespan)



async def handle_request(request: Request, api_path: str):
    token = request.headers.get('Authorization').split("Bearer ")[1] if 'Authorization' in request.headers else None
    body = await request.json()
    model = body.get('model')
    stream = body.get('stream', False)

    # Retrieve model data including the target model name
    model_data = models.get_model_data(model, api_path)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not supported for this API")
    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")
    # log model, api path and stream with timestamp as json
    logger.info({"provider": model_data.get('provider'), "model_requested": model_data.get('model_requested', model), "model_used":model_data.get('target_model', model), "api_path": api_path, "as_stream": stream})

    # Add the API key for external models
    token = model_data.get('api_key', token)

    # Replace the model name in the request body with the target model name
    body['model'] = model_data['target_model_name']

    target_url = model_data['target_base_url'] + '/' + api_path
    max_response_time = model_data.get('request_timeout', 60)
    custom_timeout = httpx.Timeout(10.0, connect=max_response_time, read=max_response_time, pool=max_response_time)

    client = httpx.AsyncClient(timeout=custom_timeout)  # Move the client outside the context
    if stream:
        # Streaming response setup
        async def stream_generator():
            async with client.stream("POST", target_url, json=body,
                                     headers={"Authorization": f"Bearer {token}"}) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
            await client.aclose()  # Close the client after streaming is complete

        return StreamingResponse(stream_generator(), media_type="application/json")

    else:
        # Non-streaming response
        response = await client.post(target_url, json=body, headers={"Authorization": f"Bearer {token}"})
        await client.aclose()  # Ensure the client is closed after request
        return response


# Modified handle_file_upload to accept multiple files
async def handle_file_upload(
    request: Request,
    api_path: str,
    files_data: Dict[str, Optional[UploadFile]],
    data: Dict[str, Optional[Any]]
):
    token = request.headers.get('Authorization')
    if token:
        token = token.split("Bearer ")[1]
    else:
        token = None
    model = data.get('model')
    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")

    model_data = models.get_model_data(model, api_path)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not supported for this API")
    logger.info({"provider": model_data.get('provider'), "model_requested": model_data.get('model_requested', model), "model_used":model_data.get('target_model', model), "api_path": api_path})
    # Update model name to target model name for backend compatibility
    data['model'] = model_data['target_model_name']

    # Prepare the files dictionary for the file upload
    files = {}
    for field_name, upload_file in files_data.items():
        if upload_file is not None:
            file_content = await upload_file.read()
            files[field_name] = (upload_file.filename, file_content, upload_file.content_type)

    # Prepare form fields as non-file data
    form_fields = {key: str(value) for key, value in data.items() if value is not None}

    target_url = f"{model_data['target_base_url']}/{api_path}"
    token = model_data.get('api_key', token)
    max_response_time = model_data.get('request_timeout', 60)
    custom_timeout = httpx.Timeout(10.0, connect=max_response_time, read=max_response_time, pool=max_response_time)

    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        # Send the request with form data and files
        response = await client.post(
            target_url,
            data=form_fields,
            files=files,
            headers={"Authorization": f"Bearer {token}"}
        )

    return response


def process_response(response, response_format):
    if response.status_code == 200:
        if response_format == 'json':
            return JSONResponse(content=response.json(), headers={'Content-Type': 'application/json'})
        else:
            return StreamingResponse(response.iter_bytes(), media_type=response.headers.get('Content-Type'))
    else:
        return JSONResponse(status_code=response.status_code, content={"message": "Failed to process request"})


def process_completion_response(response):
    # If `handle_request` returned a `StreamingResponse`, pass it directly
    if isinstance(response, StreamingResponse):
        headers = dict(response.headers)
        headers['Content-Type'] = 'text/event-stream'  # Ensure the content type for streaming responses
        response.headers.update(headers)
        return response  # Return the streaming response directly

    else:
        # Await the async JSON response and handle errors
        if response.status_code == 200:
            content = response.json()  # Properly await JSON parsing
            final_response = JSONResponse(content=content, headers={'Content-Type': 'application/json'})
            return final_response
        else:
            return JSONResponse(status_code=response.status_code, content={"message": "Failed to process request"})

# Define endpoints using the helper function for all required APIs
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    response = await handle_request(request, "v1/chat/completions")
    return process_completion_response(response)


@app.post("/v1/completions")
async def completions(request: Request):
    response = await handle_request(request, "v1/completions")
    return process_completion_response(response)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    response = await handle_request(request, "v1/embeddings")
    return response.json()


# Additional endpoints for audio, images, models, and moderation will follow the same pattern

# Define additional routes for audio services
@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    # Using the helper function to handle request forwarding and response handling
    response = await handle_request(request, "v1/audio/speech")

    # Check response status and content type before parsing
    if response.status_code == 200:
        # As it's an audio file content, it will likely not be JSON but a binary stream
        content_type = response.headers.get('Content-Type', '')

        # Streaming the audio content directly if it's a binary type (assuming mp3 or similar)
        if 'audio/' in content_type:
            return StreamingResponse(response.iter_bytes(), media_type=content_type)
        else:
            return JSONResponse(status_code=415, content={"message": "Unsupported Media Type"})
    else:
        # Log error or return a JSON message in case of failure
        return JSONResponse(status_code=response.status_code, content={"message": "Failed to generate audio"})


@app.post("/v1/audio/transcriptions")
async def audio_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form('json'),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: Optional[list] = Form(None)
):
    data = {
        'model': model,
        'language': language,
        'prompt': prompt,
        'response_format': response_format,
        'temperature': temperature,
        'timestamp_granularities': ','.join(timestamp_granularities) if timestamp_granularities else None
    }

    files_data = {
        'file': file
    }
    response = await handle_file_upload(request, "v1/audio/transcriptions", files_data, data)
    return process_response(response, response_format)


@app.post("/v1/audio/translations")
async def audio_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form('json'),
    temperature: Optional[float] = Form(0)
):
    data = {
        'model': model,
        'prompt': prompt,
        'response_format': response_format,
        'temperature': temperature
    }

    file_data = {
        'file': file
    }
    response = await handle_file_upload(request, "v1/audio/translations", file_data, data)
    return process_response(response, response_format)


# Define routes for image services
@app.post("/v1/images/generations")
async def images_generations(request: Request):
    req_body = await request.json()
    response_format = req_body.get('response_format', 'url')
    model = req_body.get('model')
    response = await handle_request(request, "v1/images/generations")
    if response_format == 'url':
        content = response.json()
        if content.get('data')[0].get('url').startswith(request.app.base_url):
            for image in content.get('data', []):  # Changed 'images' to 'data'
                url = image.get('url')
                if url:
                    parts = url.split('/')
                    # Inject the model from the request into the URL to change "https://someurl.dev/v1/images/data/1234" to "https://someurl.dev/v1/images/data/{model}/1234"
                    new_url = '/'.join(parts[:-1] + [model, parts[-1]])
                    image['url'] = new_url
        return JSONResponse(content=content)
    else:
        return response.json()


@app.post("/v1/images/edits")
async def images_edits(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    n: int = Form(1),
    response_format: str = Form('url'),
    model: str = Form(...),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),  # Ignored
    guidance_scale: Optional[float] = Form(None),  # Add-on over OpenAI
    num_inference_steps: Optional[int] = Form(None)  # Add-on over OpenAI
):
    # Prepare data dictionary
    data = {
        'prompt': prompt,
        'model': model,
        'n': n,
        'response_format': response_format,
        'size': size,
        'user': user,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps
    }

    # Prepare files dictionary
    files_data = {
        'image': image,
        'mask': mask  # May be None
    }

    response = await handle_file_upload(request, "v1/images/edits", files_data, data)
    if response_format == 'url':
        content = response.json()
        if content.get('data')[0].get('url').startswith(request.app.base_url):
            for image in content.get('data', []):  # Changed 'images' to 'data'
                url = image.get('url')
                if url:
                    parts = url.split('/')
                    # Inject the model from the request into the URL to change "https://someurl.dev/v1/images/data/1234" to "https://someurl.dev/v1/images/data/{model}/1234"
                    new_url = '/'.join(parts[:-1] + [model, parts[-1]])
                    image['url'] = new_url
        return JSONResponse(content=content)
    else:
        return response.json()

@app.post("/v1/images/variations")
async def images_variations(
    request: Request,
    image: UploadFile = File(...),
    n: int = Form(1),
    response_format: str = Form('url'),
    model: str = Form(...),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),  # Ignored
    guidance_scale: Optional[float] = Form(None),  # Add-on over OpenAI
    num_inference_steps: Optional[int] = Form(None)  # Add-on over OpenAI
):
    # Prepare data dictionary
    data = {
        'model': model,
        'n': n,
        'response_format': response_format,
        'size': size,
        'user': user,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps
    }

    # Prepare files dictionary
    files_data = {
        'image': image
    }

    response = await handle_file_upload(request, "v1/images/variations", files_data, data)
    if response_format == 'url':
        content = response.json()
        if content.get('data')[0].get('url').startswith(request.app.base_url):
            for image in content.get('data', []):  # Changed 'images' to 'data'
                url = image.get('url')
                if url:
                    parts = url.split('/')
                    # Inject the model from the request into the URL to change "https://someurl.dev/v1/images/data/1234" to "https://someurl.dev/v1/images/data/{model}/1234"
                    new_url = '/'.join(parts[:-1] + [model, parts[-1]])
                    image['url'] = new_url
        return JSONResponse(content=content)
    else:
        return response.json()




# Route for content moderation
@app.post("/v1/moderations")
async def moderations(request: Request):
    response = await handle_request(request, "v1/moderations")
    return response.json()

@app.get("/v1/images/data/{model}/{file_id}")
async def get_image_data(request: Request, model: str, file_id: str):
    token = request.headers.get('Authorization').split("Bearer ")[1] if 'Authorization' in request.headers else None

    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")

    model_data = models.get_model_data(model, "v1/images/data")
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not supported for this API")

    target_url = model_data['target_base_url'] + f'/v1/images/data/{file_id}'
    token = model_data.get('api_key', token)

    max_response_time = model_data.get('request_timeout', 60)
    custom_timeout = httpx.Timeout(10.0, connect=max_response_time, read=max_response_time, pool=max_response_time)
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        response = await client.get(target_url, headers={"Authorization": f"Bearer {token}"})

    return Response(content=response.content, media_type=response.headers.get('Content-Type'), status_code=response.status_code)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def hello_world():
    return {"msg": "hello proxy"}

@app.get("/health/{model}")
async def health_check(request: Request):
    model = request.path_params.get("model")

    model_entry = models.get_model_data(model)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not supported for this API")

    endpoints = model_entry.get("endpoints", [])

    # Collect unique target_base_url + '/health' URLs
    health_urls: Set[str] = {
        f"{endpoint['target_base_url']}/health" for endpoint in endpoints
    }

    health_timeout = next((endpoint.get("health_timeout", 10) for endpoint in endpoints), 10)

    async with httpx.AsyncClient(timeout=httpx.Timeout(health_timeout)) as client:
        results = await asyncio.gather(
            *[client.get(url) for url in health_urls],
            return_exceptions=True
        )

    all_healthy = True
    for result in results:
        if isinstance(result, Exception) or result.status_code != 200:
            all_healthy = False
            break

    if all_healthy:
        return JSONResponse(content={"status": "ok"})
    else:
        return JSONResponse(content={"status": "error"}, status_code=503)


# Route for model details
@app.get("/v1/models")
async def models_details(request: Request):
    response_data = {
        "object": "list",
        "data": models.get_model_list()
    }
    return JSONResponse(content=response_data)


@app.get("/v1/models/{model}")
async def model_details(request: Request, model: str):
    model_data = models.get_model(model)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    return JSONResponse(content=model_data)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FOAP_HOST", "0.0.0.0")
    port = int(os.getenv("FOAP_PORT", 8000))
    uvicorn.run(app, host=host, port=port)