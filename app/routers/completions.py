from fastapi import APIRouter, Request

from utils import handle_request, process_completion_response

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: Request):
    response = await handle_request(request, "v1/chat/completions")
    return process_completion_response(response)


@router.post("/completions")
async def completions(request: Request):
    response = await handle_request(request, "v1/completions")
    return process_completion_response(response)

