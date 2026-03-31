from fastapi import APIRouter, Request

from utils import handle_request, process_completion_response

router = APIRouter()


@router.post("/moderations")
async def moderations(request: Request):
    response = await handle_request(request, "v1/moderations")
    return process_completion_response(response)

