import os
from typing import Optional, List
import logging

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response

from auth import can_request
from models_handler import handler as models
from utils import handle_request, handle_file_upload, process_response, process_completion_response

__name__ = "hanseware.fast-openai-api-proxy.api_v1"

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

class FOAP_API_V1(FastAPI):
    base_url: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000")

app = FOAP_API_V1()

    # Define endpoints using the helper function for all required APIs
@app.post("/chat/completions")
async def chat_completions(request: Request):
    response = await handle_request(request, "v1/chat/completions")
    return process_completion_response(response)

@app.post("/completions")
async def completions(request: Request):
    response = await handle_request(request, "v1/completions")
    return process_completion_response(response)

@app.post("/embeddings")
async def embeddings(request: Request):
    response = await handle_request(request, "v1/embeddings")
    return response.json()

# Additional endpoints for audio, images, models, and moderation will follow the same pattern

# Define additional routes for audio services
@app.post("/audio/speech")
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

@app.post("/audio/transcriptions")
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

@app.post("/audio/translations")
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
@app.post("/images/generations")
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

@app.post("/images/edits")
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

@app.post("/images/variations")
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
@app.post("/moderations")
async def moderations(request: Request):
    response = await handle_request(request, "v1/moderations")
    return response.json()

@app.get("/images/data/{model}/{file_id}")
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

    return Response(content=response.content, media_type=response.headers.get('Content-Type'),
                    status_code=response.status_code)

# Route for model details
@app.get("/models")
async def models_details(request: Request):
    response_data = {
        "object": "list",
        "data": models.get_model_list()
    }
    return JSONResponse(content=response_data)

@app.get("/models/{model}")
async def model_details(request: Request, model: str):
    model_data = models.get_model(model)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    return JSONResponse(content=model_data)