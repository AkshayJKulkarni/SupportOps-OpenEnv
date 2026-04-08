"""Deterministic inference script for SupportOps OpenEnv using OpenAI client."""

import json
import os
import re
import time
from typing import Dict, List

try:
    import openai as Openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI package not available. Install with: pip install openai")

from app.env import SupportOpsEnv
from app.models import ActionType, TaskType, TicketAction
from app.tasks import TASK_DEFINITIONS


def get_env_config() -> Dict[str, str]:
    """Get configuration from environment variables."""
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "api_base": os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        "model_name": os.environ.get("MODEL_NAME", "gpt-4o-mini"),
    }


def build_prompt(task_definition: Dict, dialogue_history: List[str], available_actions: List[str]) -> str:
    """Build a deterministic prompt for the AI agent."""
    task = task_definition["ticket"]
    context = task_definition["context"]
    workflow_steps = task_definition["workflow_steps"]

    prompt = f"""[START]
You are a professional SaaS customer support agent. Resolve this customer ticket efficiently and effectively.

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

INSTRUCTIONS:
1. Analyze the ticket and current conversation state
2. Choose the most appropriate next action from available actions
3. Respond professionally and clearly
4. Focus on resolution efficiency
5. Use proper action types: response, escalation, request_information, apology, workaround

RESPONSE FORMAT (JSON only):
{{
    "message": "Your response message here",
    "action_type": "response",
    "step_hint": "Optional hint about next steps"
}}

Your response must be valid JSON.
[END]"""

    return prompt


def parse_ai_response(response_text: str) -> TicketAction:
    """Parse AI response into TicketAction with retry-safe error handling."""
    try:
        # Clean the response text
        cleaned = response_text.strip()

        # Remove any markdown code blocks
        cleaned = re.sub(r'```\w*\n?', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)

        # Try to parse as JSON
        response_data = json.loads(cleaned)

        # Validate required fields
        if "message" not in response_data:
            raise ValueError("Missing 'message' field in response")

        # Create TicketAction with defaults
        action = TicketAction(
            message=str(response_data["message"]),
            action_type=ActionType(response_data.get("action_type", "response")),
            step_hint=response_data.get("step_hint"),
        )

        return action

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fallback: extract message from text if JSON parsing fails
        print(f"[WARNING] Failed to parse JSON response: {type(e).__name__}")
        print(f"[WARNING] Raw response length: {len(response_text)} characters")

        # Try to extract message from plain text
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        message = " ".join(lines) if lines else "I apologize for the inconvenience. Let me investigate this issue."

        return TicketAction(
            message=message,
            action_type=ActionType.response,
            step_hint="Fallback response due to parsing error"
        )


def run_inference_for_task(task_type: TaskType, config: Dict[str, str], max_steps: int = 6) -> Dict:
    """Run inference for a single task and return results."""
    print(f"[START] Running inference for task: {task_type.value}")

    # Initialize environment
    env = SupportOpsEnv(task_type)
    task_def = TASK_DEFINITIONS[task_type]

    # Initialize OpenAI client
    client = Openai.OpenAI(
        api_key=config["api_key"],
        base_url=config["api_base"]
    )

    total_reward = 0.0
    steps_taken = 0
    final_score = 0.0

    try:
        # Reset environment
        observation = env.reset(task_type)
        print(f"[STEP] Environment reset for {task_type.value}")

        while not observation.state.done and steps_taken < max_steps:
            steps_taken += 1

            # Build prompt
            prompt = build_prompt(
                task_definition=task_def.dict(),
                dialogue_history=observation.state.dialogue_history,
                available_actions=observation.available_actions
            )

            # Get AI response
            try:
                response = client.chat.completions.create(
                    model=config["model_name"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Deterministic
                    max_tokens=512,
                )

                ai_response_text = response.choices[0].message.content.strip()

                # Parse response into action
                action = parse_ai_response(ai_response_text)

                print(f"[STEP] Step {steps_taken}: {action.action_type.value} - {action.message[:100]}...")

                # Execute step
                result = env.step(action)

                # Accumulate reward
                total_reward += result.reward.reward
                final_score = result.reward.graded_score

                # Update observation
                observation = result.observation

                # Small delay to avoid rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Step {steps_taken} failed: {e}")
                # Create a fallback action
                fallback_action = TicketAction(
                    message="I apologize for the technical difficulty. Let me escalate this to our engineering team.",
                    action_type=ActionType.escalation
                )
                result = env.step(fallback_action)
                total_reward += result.reward.reward
                observation = result.observation
                break

        # Episode complete
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
    """Main inference execution for all tasks."""
    print("[START] SupportOps OpenEnv Inference Starting")

    # Get configuration
    config = get_env_config()

    if not config["api_key"]:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")

    print(f"[INFO] Using model: {config['model_name']}")
    print(f"[INFO] API Base: {config['api_base']}")

    # Define tasks to run
    tasks = [TaskType.billing_refund, TaskType.csv_upload_bug, TaskType.sso_outage]

    # Run inference for all tasks
    results = []
    for task in tasks:
        result = run_inference_for_task(task, config)
        results.append(result)
        print()  # Empty line between tasks

    # Calculate average final score
    valid_scores = [r["final_score"] for r in results if r["final_score"] > 0]
    average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # Print summary
    print("[END] Inference completed")
    print(f"[SUMMARY] Tasks run: {len(results)}")
    print(f"[SUMMARY] Average final score: {average_score:.3f}")
    print("[SUMMARY] Individual results:")

    for result in results:
        status = "✓" if result["completed"] else "✗"
        print(f"  {status} {result['task_type']}: {result['final_score']:.3f} ({result['steps_taken']} steps)")


if __name__ == "__main__":
    main()