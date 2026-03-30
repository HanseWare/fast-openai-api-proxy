from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from models_handler import handler as models

router = APIRouter()


@router.get("/models")
async def models_details(request: Request):
    response_data = {"object": "list", "data": models.get_model_list()}
    return JSONResponse(content=response_data)


@router.get("/models/{model}")
async def model_details(request: Request, model: str):
    model_data = models.get_model(model)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    return JSONResponse(content=model_data)

