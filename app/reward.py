"""Reward shaping for the SupportOps OpenEnv tasks."""

from .models import AgentAction, TaskType
from .graders import grade_action


def compute_reward(task_id: TaskType, action: AgentAction, state: dict) -> float:
    """Compute a dense reward from grading, step count, and customer sentiment."""
    base_score = grade_action(task_id, action)
    reward = base_score * 0.8

    if state.get("steps_taken", 0) <= 1:
        reward += 0.05
    if state.get("issue_resolved"):
        reward += 0.1
    if "apolog" in action.message.lower() or "sorry" in action.message.lower():
        reward += 0.02
    if state.get("escalation_required") and "escalate" in action.message.lower():
        reward += 0.05
    if state.get("steps_taken", 0) > 5:
        reward -= 0.05

    reward = max(min(reward, 1.0), 0.0)
    return round(reward, 3)
