# SupportOps OpenEnv

SupportOps OpenEnv is a production-ready Python simulation environment for customer support training and evaluation. It includes a FastAPI backend, OpenEnv-compatible `step()`, `reset()`, and `observe()` APIs, deterministic graders, dense reward shaping, and three SaaS support tasks.

## Features

- FastAPI backend with `/reset`, `/step`, `/state`, `/health`, and `/infer`
- OpenAI-powered root inference script
- Deterministic grader scores between `0.0` and `1.0`
- Dense reward shaping for training agents
- Three realistic tasks:
  1. Billing refund request
  2. CSV upload bug escalation
  3. Enterprise SSO outage
- Docker-ready for Hugging Face Spaces

## Project Structure

- `app/`: application package
- `inference.py`: root inference client for OpenAI
- `Dockerfile`: container entrypoint
- `openenv.yaml`: OpenEnv metadata
- `requirements.txt`: Python dependencies

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API locally:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 7860
```

3. Reset the environment:

```bash
curl -X POST "http://localhost:7860/reset"
```

4. Send a step:

```bash
curl -X POST "http://localhost:7860/step" -H "Content-Type: application/json" -d '{"message": "I will investigate the billing issue and process a refund.", "action_type": "response"}'
```

## Inference

Run deterministic evaluation across all tasks:

```bash
export OPENAI_API_KEY="your-api-key"
export MODEL_NAME="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini
export API_BASE_URL="https://api.openai.com/v1"  # Optional

python inference.py
```

The script will:
- Run all 3 tasks (billing_refund, csv_upload_bug, sso_outage)
- Generate AI responses for each step
- Print `[START]`, `[STEP]`, `[END]` logs
- Calculate average final score across tasks
- Handle parsing errors gracefully

## Docker

Build and run:

```bash
docker build -t supportops-openenv .
docker run -p 7860:7860 supportops-openenv
```

## Hugging Face Spaces

The environment is Docker-ready and configured for Spaces using `openenv.yaml`.

## Notes

- Configure `OPENAI_API_KEY` to use `inference.py` or the `/infer` endpoint.
- The implementation includes TODO markers for extension and future refinement.
