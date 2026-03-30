from fastapi import APIRouter, Request

from utils import handle_request

router = APIRouter()


@router.post("/embeddings")
async def embeddings(request: Request):
    response = await handle_request(request, "v1/embeddings")
    return response.json()

