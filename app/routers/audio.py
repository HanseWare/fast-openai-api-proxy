import json
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from utils import handle_file_upload, handle_request, process_response

router = APIRouter()


def _build_transcription_payload(
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: Optional[str],
    temperature: Optional[float],
    timestamp_values: Optional[list[str]],
    include_values: Optional[list[str]],
    stream: Optional[bool],
    chunking_strategy: Optional[str],
    known_speaker_names: Optional[list[str]],
    known_speaker_references: Optional[list[str]],
) -> dict:
    effective_stream = bool(stream)

    parsed_chunking_strategy: Optional[Any] = chunking_strategy
    if chunking_strategy:
        strategy_candidate = chunking_strategy.strip()
        if strategy_candidate.startswith("{"):
            try:
                parsed_chunking_strategy = json.loads(strategy_candidate)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="chunking_strategy must be 'auto' or valid JSON.") from exc

    if timestamp_values and response_format != "verbose_json":
        raise HTTPException(
            status_code=400,
            detail="timestamp_granularities[] requires response_format='verbose_json'.",
        )

    is_diarize_model = model == "gpt-4o-transcribe-diarize"
    if is_diarize_model:
        if prompt is not None:
            raise HTTPException(status_code=400, detail="prompt is not supported for gpt-4o-transcribe-diarize.")
        if include_values:
            raise HTTPException(status_code=400, detail="include[] is not supported for gpt-4o-transcribe-diarize.")
        if timestamp_values:
            raise HTTPException(status_code=400, detail="timestamp_granularities[] is not supported for gpt-4o-transcribe-diarize.")

    if (known_speaker_names is None) != (known_speaker_references is None):
        raise HTTPException(
            status_code=400,
            detail="known_speaker_names[] and known_speaker_references[] must be provided together.",
        )

    if known_speaker_names and known_speaker_references:
        if len(known_speaker_names) != len(known_speaker_references):
            raise HTTPException(
                status_code=400,
                detail="known_speaker_names[] and known_speaker_references[] must have the same length.",
            )
        if len(known_speaker_names) > 4:
            raise HTTPException(status_code=400, detail="A maximum of 4 known speakers is supported.")

    if model == "whisper-1":
        # Whisper ignores stream mode; normalize for consistent proxy behavior.
        effective_stream = False

    return {
        "model": model,
        "language": language,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
        "timestamp_granularities[]": timestamp_values,
        "include[]": include_values,
        "stream": effective_stream,
        "chunking_strategy": parsed_chunking_strategy,
        "known_speaker_names[]": known_speaker_names,
        "known_speaker_references[]": known_speaker_references,
    }


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
    known_speaker_names_legacy: Optional[list[str]] = Form(None),
    known_speaker_references: Optional[list[str]] = Form(None, alias="known_speaker_references[]"),
    known_speaker_references_legacy: Optional[list[str]] = Form(None),
):
    timestamp_values = timestamp_granularities or timestamp_granularities_legacy
    include_values = include or include_legacy
    speaker_names = known_speaker_names or known_speaker_names_legacy
    speaker_references = known_speaker_references or known_speaker_references_legacy

    data = _build_transcription_payload(
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_values=timestamp_values,
        include_values=include_values,
        stream=stream,
        chunking_strategy=chunking_strategy,
        known_speaker_names=speaker_names,
        known_speaker_references=speaker_references,
    )

    files_data = {"file": file}
    response = await handle_file_upload(
        request,
        "v1/audio/transcriptions",
        files_data,
        data,
        stream_response=bool(data.get("stream")),
    )
    if data.get("stream"):
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
    include: Optional[list[str]] = Form(None, alias="include[]"),
    include_legacy: Optional[list[str]] = Form(None),
    stream: Optional[bool] = Form(False),
):
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

