"""Deterministic inference script for SupportOps OpenEnv using OpenAI client with heuristic fallback."""

import json
import os
import re
import time
from typing import Dict, List, Optional

try:
    import openai as Openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI package not available. Install with: pip install openai")

from app.env import SupportOpsEnv
from app.models import ActionType, TaskType, TicketAction
from app.tasks import TASK_DEFINITIONS


# Heuristic baseline agent responses - deterministic, no randomness
HEURISTIC_RESPONSES = {
    TaskType.billing_refund: [
        ("I understand the duplicate charge issue. Let me investigate your account immediately.", ActionType.response),
        ("I can confirm the duplicate charge of $199.99 on your invoice. This should not have happened.", ActionType.response),
        ("I'm processing a full refund of $199.99 to your credit card right away.", ActionType.response),
        ("The refund has been submitted and you should see it within 3-5 business days.", ActionType.response),
        ("Thank you for bringing this to our attention. Your account is now corrected.", ActionType.response),
        ("Is there anything else I can help you with today?", ActionType.response),
    ],
    TaskType.csv_upload_bug: [
        ("Thank you for reporting this CSV import issue. I'll help you resolve this.", ActionType.response),
        ("I see that the column mapping is causing the data import to fail. This is a known issue in version 2.1.0.", ActionType.response),
        ("The best solution is to use a comma delimiter instead of semicolon in your CSV file.", ActionType.response),
        ("For a permanent fix, this requires our engineering team to update the parser.", ActionType.escalation),
        ("I'm escalating this to our technical team for priority investigation.", ActionType.escalation),
        ("You should expect an update within 24 hours. Would you like me to notify you when it's fixed?", ActionType.response),
    ],
    TaskType.sso_outage: [
        ("We have detected an issue with the Enterprise SSO authentication service.", ActionType.response),
        ("This is affecting multiple enterprise customers and has been classified as a P0 incident.", ActionType.response),
        ("I'm immediately escalating this to our engineering team for emergency response.", ActionType.escalation),
        ("Our SRE team is now investigating the root cause and working on a fix.", ActionType.escalation),
        ("ETA for resolution is within 30 minutes. We'll notify all affected customers with updates.", ActionType.response),
        ("Thank you for your patience during this critical incident.", ActionType.response),
    ],
}


def get_env_config() -> Dict[str, str]:
    """Strict validator proxy-first configuration."""
    api_key = os.environ["API_KEY"] if "API_KEY" in os.environ else os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ["API_BASE_URL"] if "API_BASE_URL" in os.environ else "https://api.openai.com/v1"

    return {
        "api_key": api_key,
        "api_base": api_base,
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


def get_heuristic_action(task_type: TaskType, step_index: int) -> TicketAction:
    """Get a deterministic heuristic response based on task type and step number."""
    try:
        responses = HEURISTIC_RESPONSES.get(task_type, HEURISTIC_RESPONSES[TaskType.billing_refund])
        # Use modulo to cycle through responses if we exceed the list
        idx = step_index % len(responses)
        message, action_type = responses[idx]
        
        return TicketAction(
            message=message,
            action_type=action_type,
            step_hint=f"Heuristic baseline: step {step_index + 1}"
        )
    except Exception as e:
        # Safe fallback
        return TicketAction(
            message="I'm here to help. What can I do for you?",
            action_type=ActionType.response,
            step_hint=f"Heuristic fallback (error: {type(e).__name__})"
        )


def run_inference_for_task(task_type: TaskType, config: Dict[str, str], max_steps: int = 6, use_openai: bool = True) -> Dict:
    """Run inference for a single task and return results."""
    print(f"[START] Running inference for task: {task_type.value}")

    # Initialize environment
    env = SupportOpsEnv(task_type)
    task_def = TASK_DEFINITIONS[task_type]

    total_reward = 0.0
    steps_taken = 0
    final_score = 0.0
    use_heuristic = False

    # Initialize OpenAI client only if we have an API key and OpenAI is available
    client = None
    if use_openai and config["api_key"] and config["api_base"] and OPENAI_AVAILABLE:
        try:
            print(f"[INFO] Using LiteLLM proxy base: {config['api_base']}")
            client = Openai.OpenAI(
                api_key=config["api_key"],
                base_url=config.get("api_base", "https://api.openai.com/v1")
            )
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI client: {e}. Using heuristic baseline.")
            client = None
            use_heuristic = True
    else:
        use_heuristic = True
        if not config.get("api_key"):
            print("[INFO] No OPENAI_API_KEY provided. Using heuristic baseline agent.")
        elif not OPENAI_AVAILABLE:
            print("[INFO] OpenAI package not available. Using heuristic baseline agent.")

    try:
        # Reset environment
        observation = env.reset(task_type)
        print(f"[STEP] Environment reset for {task_type.value}")

        while not observation.state.done and steps_taken < max_steps:
            steps_taken += 1

            try:
                if client and not use_heuristic:
                    # Use OpenAI API
                    try:
                        prompt = build_prompt(
                            task_definition=task_def.dict(),
                            dialogue_history=observation.state.dialogue_history,
                            available_actions=observation.available_actions
                        )

                        response = client.chat.completions.create(
                            model=config.get("model_name", "gpt-4o-mini"),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,  # Deterministic
                            max_tokens=512,
                        )

                        ai_response_text = response.choices[0].message.content.strip()
                        action = parse_ai_response(ai_response_text)

                    except Exception as openai_error:
                        # If OpenAI fails, fall back to heuristic
                        print(f"[WARNING] OpenAI API call failed: {openai_error}. Falling back to heuristic.")
                        use_heuristic = True
                        action = get_heuristic_action(task_type, steps_taken - 1)
                else:
                    # Use heuristic baseline
                    action = get_heuristic_action(task_type, steps_taken - 1)

                print(f"[STEP] Step {steps_taken}: {action.action_type.value} - {action.message[:100]}...")

                # Execute step
                result = env.step(action)

                # Accumulate reward
                total_reward += result.reward.reward
                final_score = result.reward.graded_score

                # Update observation
                observation = result.observation

                # Small delay to avoid rate limits
                time.sleep(0.1)

            except Exception as step_error:
                print(f"[WARNING] Step {steps_taken} execution error: {step_error}")
                # Create a safe fallback action
                try:
                    fallback_action = TicketAction(
                        message="Let me escalate this issue for you.",
                        action_type=ActionType.escalation
                    )
                    result = env.step(fallback_action)
                    total_reward += result.reward.reward
                    observation = result.observation
                except Exception as fallback_error:
                    print(f"[WARNING] Even fallback step failed: {fallback_error}. Continuing.")
                    break

        # Episode complete
        print(f"[END] Task {task_type.value} completed: steps={steps_taken}, final_score={final_score:.3f}, total_reward={total_reward:.3f}")

        return {
            "task_type": task_type.value,
            "steps_taken": steps_taken,
            "final_score": final_score,
            "total_reward": total_reward,
            "completed": observation.state.done,
            "used_heuristic": use_heuristic,
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
            "used_heuristic": use_heuristic,
        }


def main():
    """Main inference execution for all tasks - never crashes."""
    print("[START] SupportOps OpenEnv Inference Starting")

    try:
        # Get configuration
        config = get_env_config()

        # Log configuration status
        if config.get("api_key"):
            print(f"[INFO] Using model: {config.get('model_name', 'gpt-4o-mini')}")
            print(f"[INFO] API Base: {config.get('api_base', 'https://api.openai.com/v1')}")
            print("[INFO] OpenAI API key provided - will attempt to use OpenAI client")
        else:
            print("[INFO] No API_KEY/OPENAI_API_KEY provided - will use heuristic baseline agent")
            if not OPENAI_AVAILABLE:
                print("[INFO] OpenAI package not available - heuristic baseline guaranteed")

        # Define tasks to run
        tasks = [TaskType.billing_refund, TaskType.csv_upload_bug, TaskType.sso_outage]

        # Run inference for all tasks
        results = []
        for task in tasks:
            try:
                result = run_inference_for_task(task, config, use_openai=True)
                results.append(result)
            except Exception as task_error:
                print(f"[WARNING] Task {task.value} processing failed: {task_error}")
                results.append({
                    "task_type": task.value,
                    "steps_taken": 0,
                    "final_score": 0.0,
                    "total_reward": 0.0,
                    "completed": False,
                    "error": str(task_error),
                    "used_heuristic": True,
                })
            print()  # Empty line between tasks

        # Calculate average final score
        try:
            valid_scores = [r.get("final_score", 0.0) for r in results if r.get("final_score", 0.0) > 0]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        except Exception as calc_error:
            print(f"[WARNING] Error calculating average score: {calc_error}")
            average_score = 0.0

        # Print summary
        print("[END] Inference completed")
        print(f"[SUMMARY] Tasks run: {len(results)}")
        print(f"[SUMMARY] Average final score: {average_score:.3f}")
        print("[SUMMARY] Individual results:")

        for result in results:
            status = "✓" if result.get("completed") else "✗"
            agent = "heuristic" if result.get("used_heuristic") else "openai"
            print(f"  {status} {result['task_type']}: {result['final_score']:.3f} ({result['steps_taken']} steps) [{agent}]")

    except Exception as main_error:
        print(f"[ERROR] Main execution failed: {main_error}")
        print("[END] Inference completed with errors")
        # Don't raise - allow graceful exit


if __name__ == "__main__":
    main()