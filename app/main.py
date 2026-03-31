import asyncio
from contextlib import asynccontextmanager
import os
from typing import Any, List, Set
import logging
from pythonjsonlogger.json import JsonFormatter
from utils import RESERVED_ATTRS

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from models_handler import handler as models
from config import is_access_control_enabled, is_admin_api_enabled, is_self_service_api_enabled
from middleware.access_control import AccessControlMiddleware
from routers.admin import router as admin_router
from routers.self_service import router as self_service_router
from api_v1 import app as api_v1_app
__name__ = "hanseware.fast-openai-api-proxy"

logger = logging.getLogger(__name__)


class FOAP(FastAPI):
    base_url: str
    model_handler: Any
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_handler = models
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000")


def setup_logging():
    loglevel = os.getenv("FOAP_LOGLEVEL", "INFO").upper()
    logger.info("Setting log level from env to", loglevel)
    logging.basicConfig(level=logging.getLevelName(loglevel))
    logHandler = logging.StreamHandler()
    formatter = JsonFormatter(timestamp=True, reserved_attrs=RESERVED_ATTRS, datefmt='%Y-%m-%d %H:%M:%S')
    logHandler.setFormatter(formatter)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logHandler)
    uvi_logger = logging.getLogger("uvicorn.access")
    uvi_logger.handlers.clear()
    uvi_logger.addHandler(logHandler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FOAP(lifespan=lifespan)

if is_access_control_enabled():
    logger.info("Enabling access control middleware")
    app.add_middleware(AccessControlMiddleware)






@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def hello_world():
    return {"msg": "hello proxy"}

@app.get("/health/{model}")
async def health_check(request: Request):
    model = request.path_params.get("model")

    model_entry = models.get_model_data(model)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not supported for this API")

    endpoints = model_entry.get("endpoints", [])

    # Collect unique target_base_url + '/health' URLs
    health_urls: Set[str] = {
        f"{endpoint['target_base_url']}/health" for endpoint in endpoints
    }

    health_timeout = next((endpoint.get("health_timeout", 10) for endpoint in endpoints), 10)

    async with httpx.AsyncClient(timeout=httpx.Timeout(health_timeout)) as client:
        results = await asyncio.gather(
            *[client.get(url) for url in health_urls],
            return_exceptions=True
        )

    all_healthy = True
    for result in results:
        if isinstance(result, Exception) or result.status_code != 200:
            all_healthy = False
            break

    if all_healthy:
        return JSONResponse(content={"status": "ok"})
    else:
        return JSONResponse(content={"status": "error"}, status_code=503)

app.mount("/v1", app=api_v1_app)

if is_admin_api_enabled():
    logger.info("Enabling admin API routes under /api/admin")
    app.include_router(admin_router)

if is_self_service_api_enabled():
    logger.info("Enabling self-service API routes under /api")
    app.include_router(self_service_router)



if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FOAP_HOST", "0.0.0.0")
    port = int(os.getenv("FOAP_PORT", 8000))
    uvicorn.run(app, host=host, port=port)