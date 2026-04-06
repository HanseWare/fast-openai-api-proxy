from fastapi import APIRouter, Request

from utils import handle_request, process_completion_response

router = APIRouter(tags=["completions"])


@router.post(
    "/chat/completions",
    summary="Create chat completion",
    description="Proxy passthrough for OpenAI-compatible chat completions, including streaming responses.",
)
async def chat_completions(request: Request):
    """POST /v1/chat/completions"""
    response = await handle_request(request, "v1/chat/completions")
    return process_completion_response(response)


@router.post(
    "/completions",
    summary="Create text completion (legacy)",
    description="Proxy passthrough for OpenAI-compatible legacy completions endpoint.",
)
async def completions(request: Request):
    """POST /v1/completions"""
    response = await handle_request(request, "v1/completions")
    return process_completion_response(response)

