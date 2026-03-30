from fastapi import APIRouter, Request

from utils import handle_request

router = APIRouter()


@router.post("/moderations")
async def moderations(request: Request):
    response = await handle_request(request, "v1/moderations")
    
    if response.status_code == 200:
        return response.json()
    
    return {"error": "Moderation request failed", "status_code": response.status_code}

