"""FastAPI server entrypoint for SupportOps OpenEnv."""

import uvicorn
from app.api import app


def main():
    """OpenEnv-compatible callable server entrypoint."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()


# Re-export the FastAPI app
__all__ = ["app", "main"]