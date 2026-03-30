from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from utils import handle_file_upload, handle_request, process_response

router = APIRouter()


@router.post("/audio/speech")
async def audio_speech(request: Request):
    response = await handle_request(request, "v1/audio/speech")

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "audio/" in content_type:
            return StreamingResponse(response.iter_bytes(), media_type=content_type)
        return JSONResponse(status_code=415, content={"message": "Unsupported Media Type"})

    return JSONResponse(status_code=response.status_code, content={"message": "Failed to generate audio"})


@router.post("/audio/transcriptions")
async def audio_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: Optional[list] = Form(None),
):
    data = {
        "model": model,
        "language": language,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
        "timestamp_granularities": ",".join(timestamp_granularities) if timestamp_granularities else None,
    }

    files_data = {"file": file}
    response = await handle_file_upload(request, "v1/audio/transcriptions", files_data, data)
    return process_response(response, response_format)


@router.post("/audio/translations")
async def audio_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
):
    data = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
    }

    files_data = {"file": file}
    response = await handle_file_upload(request, "v1/audio/translations", files_data, data)
    return process_response(response, response_format)

