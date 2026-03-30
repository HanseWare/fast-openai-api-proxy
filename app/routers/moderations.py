from fastapi import APIRouter, Request

from utils import handle_request

router = APIRouter()


@router.post("/moderations")
async def moderations(request: Request):
    response = await handle_request(request, "v1/moderations")
    return response.json()

