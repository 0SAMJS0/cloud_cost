"""GET /health — deployment liveness + model-load check."""

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    store = request.app.state.models
    return HealthResponse(
        status="ok" if store.loaded else "degraded",
        models_loaded=store.loaded,
        models=store.summary(),
    )
