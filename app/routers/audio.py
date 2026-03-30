from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from utils import handle_file_upload, handle_request, process_response

router = APIRouter()


@router.post("/audio/speech")
async def audio_speech(request: Request):
    response = await handle_request(request, "v1/audio/speech")

    if isinstance(response, StreamingResponse):
        return response

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "audio/" in content_type or "application/octet-stream" in content_type:
            return StreamingResponse(response.iter_bytes(), media_type=content_type)
        if "text/event-stream" in content_type:
            return StreamingResponse(response.iter_bytes(), media_type="text/event-stream")
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
    timestamp_granularities: Optional[list[str]] = Form(None, alias="timestamp_granularities[]"),
    timestamp_granularities_legacy: Optional[list[str]] = Form(None),
    include: Optional[list[str]] = Form(None, alias="include[]"),
    include_legacy: Optional[list[str]] = Form(None),
    stream: Optional[bool] = Form(False),
    chunking_strategy: Optional[str] = Form(None),
    known_speaker_names: Optional[list[str]] = Form(None, alias="known_speaker_names[]"),
    known_speaker_references: Optional[list[str]] = Form(None, alias="known_speaker_references[]"),
):
    timestamp_values = timestamp_granularities or timestamp_granularities_legacy
    include_values = include or include_legacy

    data = {
        "model": model,
        "language": language,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
        "timestamp_granularities[]": timestamp_values,
        "include[]": include_values,
        "stream": stream,
        "chunking_strategy": chunking_strategy,
        "known_speaker_names[]": known_speaker_names,
        "known_speaker_references[]": known_speaker_references,
    }

    files_data = {"file": file}
    response = await handle_file_upload(
        request,
        "v1/audio/transcriptions",
        files_data,
        data,
        stream_response=bool(stream),
    )
    if stream:
        return response

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
    include: Optional[list[str]] = Form(None, alias="include[]"),
    include_legacy: Optional[list[str]] = Form(None),
    stream: Optional[bool] = Form(False),

    include_values = include or include_legacy

    data = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
        "include[]": include_values,
        "stream": stream,
    }

    files_data = {"file": file}
    response = await handle_file_upload(
        request,
        "v1/audio/translations",
        files_data,
        data,
        stream_response=bool(stream),
    )
    if stream:
        return response

    return process_response(response, response_format)

