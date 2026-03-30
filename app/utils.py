import logging
import json
from typing import Dict, Optional, Any, List

import httpx
from fastapi import Request, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, Response

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


def _extract_bearer_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or "Bearer " not in auth_header:
        return None
    token = auth_header.split("Bearer ", 1)[1].strip()
    return token or None


def _form_value_to_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)

async def handle_request(request: Request, api_path: str):
    token = _extract_bearer_token(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    model = body.get('model')
    stream = body.get('stream', False) or body.get('stream_format') == 'sse'

    if not model:
        raise HTTPException(status_code=400, detail="Missing model in request body")

    # Retrieve model data including the target model name
    model_data = models.get_model_data(model, api_path)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not supported for this API")
    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")
    # log model, api path and stream with timestamp as json
    logger.info({"provider": model_data.get('provider'), "model_requested": model_data.get('model_requested', model), "model_used":model_data.get('target_model_name', model), "api_path": api_path, "as_stream": stream})

    # Add the API key for external models
    token = model_data.get('api_key', token)

    # Replace the model name in the request body with the target model name
    body['model'] = model_data['target_model_name']

    target_url = model_data['target_base_url'] + '/' + api_path
    max_response_time = model_data.get('request_timeout', 60)
    custom_timeout = httpx.Timeout(max_response_time, connect=max_response_time, read=max_response_time, pool=max_response_time)

    client = httpx.AsyncClient(timeout=custom_timeout)  # Move the client outside the context
    if stream:
        # Streaming response setup
        async def stream_generator():
            async with client.stream("POST", target_url, json=body,
                                     headers={"Authorization": f"Bearer {token}"}) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
            await client.aclose()  # Close the client after streaming is complete

        media_type = "text/event-stream" if body.get('stream', False) or body.get('stream_format') == 'sse' else "application/json"
        return StreamingResponse(stream_generator(), media_type=media_type)

    else:
        # Non-streaming response
        try:
            response = await client.post(target_url, json=body, headers={"Authorization": f"Bearer {token}"})
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc
        await client.aclose()  # Ensure the client is closed after request
        return response


async def handle_subresource_request(
    request: Request,
    api_path: str,
    target_suffix: str,
    method: str,
    allow_stream: bool = False,
):
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    method_upper = method.upper()
    body: Dict[str, Any] = {}
    model = request.query_params.get("model")
    stream = False

    if method_upper != "GET":
        raw_body = await request.body()
        if raw_body:
            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        model = model or body.get("model")
        stream = allow_stream and (body.get("stream", False) or body.get("stream_format") == "sse")

    if not model:
        raise HTTPException(status_code=400, detail="Missing model. Provide ?model=... or include model in body.")
    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")

    model_data = models.get_model_data(model, api_path)
    logger.info({
        "provider": model_data.get("provider"),
        "model_requested": model_data.get("model_requested", model),
        "model_used": model_data.get("target_model_name", model),
        "api_path": f"{api_path}{target_suffix}",
        "method": method_upper,
        "as_stream": stream,
    })

    token = model_data.get("api_key", token)
    if "model" in body and model_data.get("target_model_name"):
        body["model"] = model_data["target_model_name"]

    target_url = f"{model_data['target_base_url']}/{api_path}{target_suffix}"
    max_response_time = model_data.get("request_timeout", 60)
    custom_timeout = httpx.Timeout(max_response_time, connect=max_response_time, read=max_response_time, pool=max_response_time)
    client = httpx.AsyncClient(timeout=custom_timeout)

    if stream:
        async def stream_generator():
            async with client.stream(
                method_upper,
                target_url,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
            await client.aclose()

        media_type = "text/event-stream" if body.get("stream") or body.get("stream_format") == "sse" else "application/json"
        return StreamingResponse(stream_generator(), media_type=media_type)

    try:
        if method_upper == "GET":
            response = await client.get(target_url, headers={"Authorization": f"Bearer {token}"})
        elif method_upper == "POST":
            response = await client.post(target_url, json=body, headers={"Authorization": f"Bearer {token}"})
        elif method_upper == "DELETE":
            request_kwargs = {"headers": {"Authorization": f"Bearer {token}"}}
            if body:
                request_kwargs["json"] = body
            response = await client.delete(target_url, **request_kwargs)
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported passthrough method: {method_upper}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc
    finally:
        await client.aclose()

    return response


# Modified handle_file_upload to accept multiple files
async def handle_file_upload(
    request: Request,
    api_path: str,
    files_data: Dict[str, Optional[UploadFile]],
    data: Dict[str, Optional[Any]],
    stream_response: bool = False,
):
    token = _extract_bearer_token(request)
    model = data.get('model')
    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")

    model_data = models.get_model_data(model, api_path)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not supported for this API")
    logger.info({"provider": model_data.get('provider'), "model_requested": model_data.get('model_requested', model), "model_used":model_data.get('target_model_name', model), "api_path": api_path})
    # Update model name to target model name for backend compatibility
    data['model'] = model_data['target_model_name']

    # Prepare the files dictionary for the file upload
    files = {}
    for field_name, upload_file in files_data.items():
        if upload_file is not None:
            file_content = await upload_file.read()
            files[field_name] = (upload_file.filename, file_content, upload_file.content_type)

    # Preserve repeated form fields for keys like include[] / timestamp_granularities[].
    form_fields = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    form_fields.append((key, _form_value_to_string(item)))
        else:
            form_fields.append((key, _form_value_to_string(value)))

    target_url = f"{model_data['target_base_url']}/{api_path}"
    token = model_data.get('api_key', token)
    max_response_time = model_data.get('request_timeout', 60)
    custom_timeout = httpx.Timeout(10.0, connect=max_response_time, read=max_response_time, pool=max_response_time)

    if stream_response:
        client = httpx.AsyncClient(timeout=custom_timeout)

        async def stream_generator():
            async with client.stream(
                "POST",
                target_url,
                data=form_fields,
                files=files,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
            await client.aclose()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            response = await client.post(
                target_url,
                data=form_fields,
                files=files,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    return response


def process_response(response, response_format):
    content_type = response.headers.get("Content-Type", "")

    if response.status_code == 200:
        if response_format == "json":
            try:
                payload = response.json()
            except Exception:
                payload = {"message": response.text or "Invalid JSON response from upstream"}
                return JSONResponse(status_code=502, content=payload)
            return JSONResponse(content=payload, headers={"Content-Type": "application/json"})
        return Response(content=response.content, media_type=content_type or "application/octet-stream")

    if "application/json" in content_type:
        try:
            error_content = response.json()
        except Exception:
            error_content = {"message": "Failed to process request", "status_code": response.status_code}
        return JSONResponse(status_code=response.status_code, content=error_content)

    return JSONResponse(
        status_code=response.status_code,
        content={"message": response.text or "Failed to process request", "status_code": response.status_code},
    )


def process_completion_response(response):
    # If `handle_request` returned a `StreamingResponse`, pass it directly
    if isinstance(response, StreamingResponse):
        if not response.headers.get("Content-Type"):
            response.headers["Content-Type"] = "text/event-stream"
        return response  # Return the streaming response directly

    # Keep backend payload and status codes for non-streaming responses.
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except Exception:
            payload = {"message": response.text or "Invalid JSON response from upstream"}
        return JSONResponse(status_code=response.status_code, content=payload, headers={"Content-Type": "application/json"})

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type or "application/octet-stream",
    )
