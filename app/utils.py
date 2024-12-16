import logging
from typing import Dict, Optional, Any, List

import httpx
from fastapi import Request, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse

from auth import can_request
from models_handler import handler as models

__name__ = "hanseware.fast-openai-api-proxy.utils"

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
