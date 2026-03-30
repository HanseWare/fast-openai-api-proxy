import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

__name__ = "hanseware.fast-openai-api-proxy.api_v1.fallback"

logger = logging.getLogger(__name__)
router = APIRouter()


@router.api_route("/{path:path}", methods=["GET", "POST"])
async def fallback_route(path: str, request: Request):
    full_url = str(request.url)
    logger.info("---------------------------------------------------------------------------------")
    logger.info(f"Fallback route triggered for {request.method} {full_url}")

    body = await request.body()
    logger.info(f"Request body: {body.decode('utf-8', errors='replace')}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request query parameters: {dict(request.query_params)}")
    logger.info(f"Request client IP: {request.client.host if request.client else 'unknown'}")
    logger.info(f"Request path: {path}")
    logger.info("---------------------------------------------------------------------------------")

    return JSONResponse(status_code=404, content={"detail": "Nothing to see here!"})

