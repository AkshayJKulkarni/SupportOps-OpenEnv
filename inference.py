"""Deterministic inference script for SupportOps OpenEnv using OpenAI client."""

import json
import os
import re
import sys
import time
from typing import Dict, List

from openai import OpenAI

from app.env import SupportOpsEnv
from app.models import ActionType, TaskType, TicketAction
from app.tasks import TASK_DEFINITIONS


def get_env_config() -> Dict[str, str]:
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        raise EnvironmentError("API_KEY or OPENAI_API_KEY environment variable is required")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model_name": os.environ.get("MODEL_NAME", "gpt-4o-mini"),
    }


def build_prompt(task_definition: Dict, dialogue_history: List[str], available_actions: List[str]) -> str:
    task = task_definition["ticket"]
    context = task_definition["context"]
    workflow_steps = task_definition["workflow_steps"]

    return f"""You are a professional SaaS customer support agent. Resolve this customer ticket efficiently.

TASK INFORMATION:
- Title: {task['title']}
- Description: {task['description']}
- Customer: {task['customer_name']} ({task['account_tier']} tier)
- Priority: {task_definition['priority']}
- Context: {context}

EXPECTED WORKFLOW:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(workflow_steps))}

CURRENT CONVERSATION:
{chr(10).join(f"Agent: {msg[:200]}" for msg in dialogue_history) if dialogue_history else "No previous messages."}

AVAILABLE ACTIONS:
{chr(10).join(f"- {action}" for action in available_actions)}

RESPONSE FORMAT (JSON only):
{{
    "message": "Your response message here",
    "action_type": "response",
    "step_hint": "Optional hint about next steps"
}}"""


def parse_ai_response(response_text: str) -> TicketAction:
    try:
        cleaned = re.sub(r'```\w*\n?', '', response_text.strip())
        cleaned = re.sub(r'```', '', cleaned)
        response_data = json.loads(cleaned)

        if "message" not in response_data:
            raise ValueError("Missing 'message' field in response")

        return TicketAction(
            message=str(response_data["message"]),
            action_type=ActionType(response_data.get("action_type", "response")),
            step_hint=response_data.get("step_hint"),
        )

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[WARNING] Failed to parse JSON response: {type(e).__name__}")
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        return TicketAction(
            message=" ".join(lines) if lines else "I apologize for the inconvenience. Let me investigate this issue.",
            action_type=ActionType.response,
            step_hint="Fallback response due to parsing error",
        )


def run_inference_for_task(task_type: TaskType, config: Dict[str, str], max_steps: int = 6) -> Dict:
    print(f"[START] Running inference for task: {task_type.value}")

    env = SupportOpsEnv(task_type)
    task_def = TASK_DEFINITIONS[task_type]
    try:
        client = OpenAI(api_key=config["api_key"], base_url=config["api_base"])
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}")
        raise

    total_reward = 0.0
    steps_taken = 0
    final_score = 0.0

    try:
        observation = env.reset(task_type)
        print(f"[STEP] Environment reset for {task_type.value}")

        while not observation.state.done and steps_taken < max_steps:
            steps_taken += 1

            try:
                prompt = build_prompt(
                    task_definition=task_def.model_dump(mode="json"),
                    dialogue_history=observation.state.dialogue_history,
                    available_actions=observation.available_actions,
                )

                response = client.chat.completions.create(
                    model=config["model_name"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )

                action = parse_ai_response(response.choices[0].message.content.strip())
                print(f"[STEP] Step {steps_taken}: {action.action_type.value} - {action.message[:100]}...")

                result = env.step(action)
                total_reward += result.reward.reward
                final_score = result.reward.graded_score
                observation = result.observation
                time.sleep(0.1)

            except Exception as step_error:
                print(f"[WARNING] Step {steps_taken} failed: {step_error}")
                try:
                    result = env.step(TicketAction(
                        message="Let me escalate this issue for you.",
                        action_type=ActionType.escalation,
                    ))
                    total_reward += result.reward.reward
                    observation = result.observation
                except Exception:
                    break

        print(f"[END] Task {task_type.value} completed: steps={steps_taken}, final_score={final_score:.3f}, total_reward={total_reward:.3f}")
        return {
            "task_type": task_type.value,
            "steps_taken": steps_taken,
            "final_score": final_score,
            "total_reward": total_reward,
            "completed": observation.state.done,
        }

    except Exception as e:
        print(f"[ERROR] Task {task_type.value} failed: {e}")
        return {
            "task_type": task_type.value,
            "steps_taken": steps_taken,
            "final_score": 0.0,
            "total_reward": total_reward,
            "completed": False,
            "error": str(e),
        }


def main():
    print("[START] SupportOps OpenEnv Inference Starting")

    config = get_env_config()
    print(f"[INFO] Using model: {config['model_name']}")
    print(f"[INFO] API Base: {config['api_base']}")

    tasks = [TaskType.billing_refund, TaskType.csv_upload_bug, TaskType.sso_outage]
    results = []

    for task in tasks:
        results.append(run_inference_for_task(task, config))
        print()

    valid_scores = [r["final_score"] for r in results if r["final_score"] > 0]
    average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    print("[END] Inference completed")
    print(f"[SUMMARY] Tasks run: {len(results)}")
    print(f"[SUMMARY] Average final score: {average_score:.3f}")
    for result in results:
        status = "✓" if result.get("completed") else "✗"
        print(f"  {status} {result['task_type']}: {result['final_score']:.3f} ({result['steps_taken']} steps)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Fatal: {e}")
    sys.exit(0)
