import os

from fastapi import FastAPI

from routers.audio import router as audio_router
from routers.completions import router as completions_router
from routers.embeddings import router as embeddings_router
from routers.fallback import router as fallback_router
from routers.images import router as images_router
from routers.models import router as models_router
from routers.moderations import router as moderations_router
from routers.responses import router as responses_router


class FOAP_API_V1(FastAPI):
    base_url: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000")


app = FOAP_API_V1()

# Include fallback router last so specific routes always win.
app.include_router(completions_router)
app.include_router(embeddings_router)
app.include_router(audio_router)
app.include_router(images_router)
app.include_router(moderations_router)
app.include_router(responses_router)
app.include_router(models_router)
app.include_router(fallback_router)
