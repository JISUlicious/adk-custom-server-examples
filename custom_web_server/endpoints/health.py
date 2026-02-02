"""
Health check endpoints.
"""

from fastapi import APIRouter


def create_health_router() -> APIRouter:
    """Create health check router."""
    router = APIRouter(tags=["Health"])

    @router.get("/health")
    async def health_check():
        """Basic health check."""
        return {"status": "healthy"}

    @router.get("/ready")
    async def readiness_check():
        """Readiness check."""
        return {"status": "ready"}

    return router
