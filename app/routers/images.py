from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from auth import can_request
from models_handler import handler as models
from utils import handle_file_upload, handle_request, process_completion_response

router = APIRouter()


def _patch_image_data_urls(request: Request, content: dict, model: str) -> dict:
    data = content.get("data") or []
    if not data:
        return content

    first_url = data[0].get("url") if isinstance(data[0], dict) else None
    if not first_url or not first_url.startswith(request.app.base_url):
        return content

    for image in data:
        url = image.get("url")
        if not url:
            continue
        parts = url.split("/")
        image["url"] = "/".join(parts[:-1] + [model, parts[-1]])

    return content


@router.post("/images/generations")
async def images_generations(request: Request):
    req_body = await request.json()
    response_format = req_body.get("response_format", "url")
    model = req_body.get("model")

    response = await handle_request(request, "v1/images/generations")
    
    # Handle streaming response if stream=true was requested
    if isinstance(response, StreamingResponse):
        return response

    if response.status_code != 200:
        return process_completion_response(response)
    
    if response_format == "url":
        content = _patch_image_data_urls(request, response.json(), model)
        return JSONResponse(content=content)

    return response.json()


@router.post("/images/edits")
async def images_edits(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    n: int = Form(1),
    response_format: str = Form("url"),
    model: str = Form(...),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    background: Optional[str] = Form(None),
    moderation: Optional[str] = Form(None),
    compression: Optional[int] = Form(None),
    stream: Optional[bool] = Form(False),
    partial_images: Optional[int] = Form(None),
):
    data = {
        "prompt": prompt,
        "model": model,
        "n": n,
        "response_format": response_format,
        "size": size,
        "user": user,
        "quality": quality,
        "background": background,
        "moderation": moderation,
        "compression": compression,
        "stream": stream,
        "partial_images": partial_images,
    }

    files_data = {"image": image, "mask": mask}

    response = await handle_file_upload(
        request,
        "v1/images/edits",
        files_data,
        data,
        stream_response=bool(stream),
    )
    if stream:
        return response

    if response.status_code != 200:
        return process_completion_response(response)

    if response_format == "url":
        content = _patch_image_data_urls(request, response.json(), model)
        return JSONResponse(content=content)

    return response.json()


@router.post("/images/variations")
async def images_variations(
    request: Request,
    image: UploadFile = File(...),
    n: int = Form(1),
    response_format: str = Form("url"),
    model: str = Form(...),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    background: Optional[str] = Form(None),
    moderation: Optional[str] = Form(None),
    compression: Optional[int] = Form(None),
    stream: Optional[bool] = Form(False),
    partial_images: Optional[int] = Form(None),
):
    data = {
        "model": model,
        "n": n,
        "response_format": response_format,
        "size": size,
        "user": user,
        "quality": quality,
        "background": background,
        "moderation": moderation,
        "compression": compression,
        "stream": stream,
        "partial_images": partial_images,
    }

    files_data = {"image": image}

    response = await handle_file_upload(
        request,
        "v1/images/variations",
        files_data,
        data,
        stream_response=bool(stream),
    )
    if stream:
        return response

    if response.status_code != 200:
        return process_completion_response(response)

    if response_format == "url":
        content = _patch_image_data_urls(request, response.json(), model)
        return JSONResponse(content=content)

    return response.json()


@router.get("/images/data/{model}/{file_id}")
async def get_image_data(request: Request, model: str, file_id: str):
    token = request.headers.get("Authorization").split("Bearer ")[1] if "Authorization" in request.headers else None

    if not can_request(model, token):
        raise HTTPException(status_code=403, detail="Unauthorized")

    model_data = models.get_model_data(model, "v1/images/data")
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not supported for this API")

    target_url = model_data["target_base_url"] + f"/v1/images/data/{file_id}"
    token = model_data.get("api_key", token)

    max_response_time = model_data.get("request_timeout", 60)
    custom_timeout = httpx.Timeout(10.0, connect=max_response_time, read=max_response_time, pool=max_response_time)
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        response = await client.get(target_url, headers={"Authorization": f"Bearer {token}"})

    return Response(
        content=response.content,
        media_type=response.headers.get("Content-Type"),
        status_code=response.status_code,
    )

