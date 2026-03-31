from fastapi import APIRouter, Request

from utils import handle_request, process_completion_response

router = APIRouter(tags=["moderations"])


@router.post(
    "/moderations",
    summary="Create moderation",
    description="Proxy passthrough for OpenAI-compatible moderation requests.",
)
async def moderations(request: Request):
    """POST /v1/moderations"""
    response = await handle_request(request, "v1/moderations")
    return process_completion_response(response)

