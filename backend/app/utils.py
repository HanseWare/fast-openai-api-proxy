import logging
import json
from typing import Dict, Optional, Any, List

import httpx
from fastapi import Request, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from access_store import store
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

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def _proxy_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


def _close_stream_resources(response: httpx.Response, client: httpx.AsyncClient) -> None:
    pass

async def _close_stream_resources_async(response: httpx.Response, client: httpx.AsyncClient) -> None:
    await response.aclose()
    await client.aclose()

def extract_provider_ratelimits(headers: httpx.Headers) -> dict:
    limits = {}
    for key, val in headers.items():
        k = key.lower()
        if k == 'x-ratelimit-limit-minute': limits['limit_minute'] = int(val)
        elif k == 'x-ratelimit-remaining-minute': limits['remaining_minute'] = int(val)
        elif k == 'x-ratelimit-limit-hour': limits['limit_hour'] = int(val)
        elif k == 'x-ratelimit-remaining-hour': limits['remaining_hour'] = int(val)
        elif k == 'x-ratelimit-limit-day': limits['limit_day'] = int(val)
        elif k == 'x-ratelimit-remaining-day': limits['remaining_day'] = int(val)
        elif k == 'ratelimit-limit': limits['limit_minute'] = int(val)
        elif k == 'ratelimit-remaining': limits['remaining_minute'] = int(val)
    return limits


def _raise_if_provider_rate_limited(model_data: Dict[str, Any]) -> None:
    if not model_data.get('sync_provider_ratelimits'):
        return

    provider_name = model_data.get('provider')
    if not provider_name:
        return

    exhausted_windows = store.get_exhausted_provider_ratelimit_windows(provider_name)
    if exhausted_windows:
        windows = ", ".join(exhausted_windows)
        raise HTTPException(
            status_code=429,
            detail=f"Upstream provider rate limit exhausted for {provider_name} ({windows})",
        )


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

    _raise_if_provider_rate_limited(model_data)

    # Replace the model name in the request body with the target model name
    body['model'] = model_data['target_model_name']

    target_url = model_data['target_base_url'] + '/' + api_path
    max_response_time = model_data.get('request_timeout', 60)
    custom_timeout = httpx.Timeout(max_response_time, connect=max_response_time, read=max_response_time, pool=max_response_time)

    client = httpx.AsyncClient(timeout=custom_timeout)
    import asyncio

    async def _try_request(url: str, tkn: str, req_body: dict):
        if stream:
            req = client.build_request("POST", url, json=req_body, headers={"Authorization": f"Bearer {tkn}"})
            return await client.send(req, stream=True)
        else:
            return await client.post(url, json=req_body, headers={"Authorization": f"Bearer {tkn}"})

    try:
        response = await _try_request(target_url, token, body)
        
        # 429 Retry logic
        if response.status_code == 429 and model_data.get('max_upstream_retry_seconds', 0) > 0:
            retry_after_str = response.headers.get("Retry-After", "1")
            try: retry_after = int(retry_after_str)
            except ValueError: retry_after = 1
            
            if retry_after <= model_data['max_upstream_retry_seconds']:
                logger.info(f"Upstream 429 hit. Retrying in {retry_after}s for {model_data['provider']}")
                if stream:
                    await response.aread()
                    await response.aclose()
                await asyncio.sleep(retry_after)
                response = await _try_request(target_url, token, body)

        # Fallback Model logic
        if response.status_code >= 400 and model_data.get('fallback_model_name'):
            fallback_data = models.get_model_data(model_data['fallback_model_name'], api_path)
            if fallback_data:
                logger.info(f"Upstream error {response.status_code}, routing to fallback: {fallback_data['target_model_name']}")
                if stream:
                    await response.aread()
                    await response.aclose()
                b2 = body.copy()
                b2['model'] = fallback_data['target_model_name']
                f_url = fallback_data['target_base_url'] + '/' + api_path
                f_token = fallback_data.get('api_key', token)
                response = await _try_request(f_url, f_token, b2)

    except httpx.RequestError as exc:
        if model_data.get('fallback_model_name'):
            fallback_data = models.get_model_data(model_data['fallback_model_name'], api_path)
            if fallback_data:
                logger.info(f"Upstream request error, routing to fallback: {fallback_data['target_model_name']}")
                b2 = body.copy()
                b2['model'] = fallback_data['target_model_name']
                f_url = fallback_data['target_base_url'] + '/' + api_path
                f_token = fallback_data.get('api_key', token)
                try:
                    response = await _try_request(f_url, f_token, b2)
                except httpx.RequestError as inner_exc:
                    await client.aclose()
                    raise HTTPException(status_code=502, detail=f"Fallback request failed: {inner_exc}") from inner_exc
            else:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc
        else:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if model_data.get('sync_provider_ratelimits'):
        limits = extract_provider_ratelimits(response.headers)
        if limits:
            from access_store import store
            store.sync_provider_ratelimits(model_data['provider'], limits)

    if stream:
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            await client.aclose()
            return response

        stream_content_type = response.headers.get("Content-Type")
        media_type = stream_content_type or "text/event-stream"
        stream_headers = _proxy_response_headers(response.headers)
        stream_headers.pop("content-type", None)

        return StreamingResponse(
            response.aiter_bytes(),
            status_code=response.status_code,
            media_type=media_type,
            headers=stream_headers,
            background=BackgroundTask(_close_stream_resources_async, response, client),
        )
    else:
        await client.aclose()
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

    _raise_if_provider_rate_limited(model_data)

    target_url = f"{model_data['target_base_url']}/{api_path}{target_suffix}"
    max_response_time = model_data.get("request_timeout", 60)
    custom_timeout = httpx.Timeout(max_response_time, connect=max_response_time, read=max_response_time, pool=max_response_time)
    client = httpx.AsyncClient(timeout=custom_timeout)

    if stream:
        request_kwargs: Dict[str, Any] = {"headers": {"Authorization": f"Bearer {token}"}}
        if method_upper != "GET" and body:
            request_kwargs["json"] = body

        try:
            upstream_request = client.build_request(method_upper, target_url, **request_kwargs)
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

        if upstream_response.status_code >= 400:
            await upstream_response.aread()
            await upstream_response.aclose()
            await client.aclose()
            return upstream_response

        stream_content_type = upstream_response.headers.get("Content-Type")
        media_type = stream_content_type or "text/event-stream"
        stream_headers = _proxy_response_headers(upstream_response.headers)
        stream_headers.pop("content-type", None)

        return StreamingResponse(
            upstream_response.aiter_bytes(),
            status_code=upstream_response.status_code,
            media_type=media_type,
            headers=stream_headers,
            background=BackgroundTask(_close_stream_resources, upstream_response, client),
        )

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

    if model_data.get('sync_provider_ratelimits') and response:
        limits = extract_provider_ratelimits(response.headers)
        if limits:
            from access_store import store
            store.sync_provider_ratelimits(model_data['provider'], limits)

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

    _raise_if_provider_rate_limited(model_data)

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
                data=form_fields,  # type: ignore[arg-type]
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
                data=form_fields,  # type: ignore[arg-type]
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
