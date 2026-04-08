"""FastAPI backend exposing SupportOps OpenEnv environment endpoints."""

import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .env import SupportOpsEnv
from .models import (
    ErrorResponse,
    HealthResponse,
    OpenAIRequest,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResponse,
    TaskType,
    TicketAction,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SupportOps OpenEnv",
    description="Customer support ticket resolution simulation environment",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for web compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global environment singleton
ENV = SupportOpsEnv()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for production error handling."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            code=500,
        ).dict(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="ok",
        message="SupportOps OpenEnv is running",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/reset", response_model=ResetResponse)
async def reset_environment(request: ResetRequest = ResetRequest()) -> ResetResponse:
    """Reset the environment to initial state for a given task.

    Args:
        request: Reset request with optional task_id

    Returns:
        ResetResponse containing initial observation and logs
    """
    try:
        task_id = request.task_id or TaskType.billing_refund
        logger.info(f"Resetting environment for task: {task_id.value}")

        observation = ENV.reset(task_id)

        return ResetResponse(
            observation=observation,
            logs=ENV.logs.copy(),
        )
    except Exception as e:
        logger.error(f"Error during reset: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Reset failed",
                detail=str(e),
                code=500,
            ).dict(),
        )


@app.post("/step", response_model=StepResponse)
async def step_environment(action: TicketAction) -> StepResponse:
    """Execute a single step in the environment.

    Args:
        action: The action to execute

    Returns:
        StepResponse containing observation, reward, done status, and logs

    Raises:
        HTTPException: If episode is already complete
    """
    try:
        if ENV.state.done:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="Episode completed",
                    detail="Environment episode already completed. Call /reset first.",
                    code=400,
                ).dict(),
            )

        logger.info(f"Executing step with action: {action.action_type.value}")

        result = ENV.step(action)

        return StepResponse(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            logs=result.logs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during step: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Step failed",
                detail=str(e),
                code=500,
            ).dict(),
        )


@app.get("/state", response_model=StateResponse)
async def get_environment_state() -> StateResponse:
    """Get the current environment state observation.

    Returns:
        StateResponse containing current observation and logs
    """
    try:
        observation = ENV.observe()

        return StateResponse(
            observation=observation,
            logs=ENV.logs.copy(),
        )
    except Exception as e:
        logger.error(f"Error getting state: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="State retrieval failed",
                detail=str(e),
                code=500,
            ).dict(),
        )


@app.post("/infer")
async def infer_with_openai(request: OpenAIRequest):
    """Inference endpoint using OpenAI client (optional feature)."""
    try:
        from openai import OpenAI

        api_key = request.__dict__.get("openai_api_key") or __import__("os").environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="API key required",
                    detail="OpenAI API key not provided",
                    code=400,
                ).dict(),
            )

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=request.model,
            input=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return {"response": response}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Inference failed",
                detail=str(e),
                code=500,
            ).dict(),
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the environment on startup."""
    logger.info("SupportOps OpenEnv API starting up")
    global ENV
    ENV = SupportOpsEnv()
    logger.info("Environment initialized successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("SupportOps OpenEnv API shutting down")
