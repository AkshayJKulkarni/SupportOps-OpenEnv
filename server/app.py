"""FastAPI server entrypoint for SupportOps OpenEnv."""

from app.api import app

# Re-export the FastAPI app for the server entrypoint
__all__ = ["app"]