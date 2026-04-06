from fastapi import APIRouter, Request

from utils import handle_request, process_completion_response

router = APIRouter(tags=["embeddings"])


@router.post(
    "/embeddings",
    summary="Create embeddings",
    description="Proxy passthrough for OpenAI-compatible embeddings requests.",
)
async def embeddings(request: Request):
    """POST /v1/embeddings"""
    response = await handle_request(request, "v1/embeddings")
    return process_completion_response(response)

