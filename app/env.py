"""OpenEnv-compatible environment implementation for SupportOps."""

from typing import List, Optional, Set
from .models import AgentAction, EnvState, Observation, RewardResult, StepResult, TaskType
from .tasks import TASK_DEFINITIONS
from .graders import grade_action
from .reward import compute_reward


class SupportOpsEnv:
    """A training environment to simulate customer support ticket resolution."""

    def __init__(self, task_id: TaskType = TaskType.billing_refund):
        self.task_id = task_id
        self.task_definition = TASK_DEFINITIONS[task_id]
        self.state = self._build_initial_state(task_id)
        self.logs: List[str] = []
        self.completed_actions: Set[str] = set()
        self.max_steps = 8  # Allow more steps for complex tasks

    def reset(self, task_id: Optional[TaskType] = None) -> Observation:
        """Reset the environment to initial state for a given task."""
        self._log("[START] Resetting environment")
        if task_id is not None:
            self.task_id = task_id
            self.task_definition = TASK_DEFINITIONS[task_id]
        self.state = self._build_initial_state(self.task_id)
        self.logs = []  # Reset logs on environment reset
        self.completed_actions = set()
        self._log("[END] Reset complete")
        return self._build_observation()

    def step(self, action: AgentAction) -> StepResult:
        """Execute a single step in the environment."""
        if self.state.done:
            raise ValueError("Environment episode is already complete. Call reset() first.")

        self._log(f"[STEP] Received action: {action.message[:100]}...")

        # Prevent repeated identical actions
        action_key = f"{action.action_type.value}:{action.message.lower().strip()}"
        if action_key in self.completed_actions:
            self._log("[STEP] Duplicate action detected, penalizing")
            return self._create_penalty_step("Repeated action detected. Please provide new information.")

        self.completed_actions.add(action_key)
        self.state.steps_taken += 1
        self.state.dialogue_history.append(action.message)

        # Compute grading and reward
        graded_score = max(0.01, min(0.99, grade_action(self.task_id, action)))
        reward_value = compute_reward(self.task_id, action, self.state.model_dump(mode="json"))
        self.state.reward_score = reward_value

        # Update resolution state based on action
        self._update_resolution_state(action)

        # Check if episode should end
        done = self._check_done()
        self.state.done = done

        reward_result = RewardResult(
            reward=reward_value,
            graded_score=graded_score,
            note=self._build_reward_note(action, graded_score, reward_value),
        )

        observation = self._build_observation()
        self._log(f"[STEP] Step complete: done={done} reward={reward_value:.3f}")
        if done:
            self._log("[END] Episode finished")
        return StepResult(
            observation=observation,
            reward=reward_result,
            done=done,
            logs=self.logs.copy(),
        )

    def observe(self) -> Observation:
        """Get current environment state observation."""
        return self._build_observation()

    def _build_initial_state(self, task_id: TaskType) -> EnvState:
        """Build the initial state for a given task."""
        ticket = self.task_definition.ticket
        return EnvState(
            ticket=ticket,
            dialogue_history=[],
            issue_resolved=False,
            escalation_required=self.task_definition.grading_metadata.get("bug_escalation") == "required",
            steps_taken=0,
            reward_score=0.01,
            done=False,
        )

    def _build_observation(self) -> Observation:
        """Build observation from current state."""
        available_actions = self._suggest_actions()
        return Observation(state=self.state, available_actions=available_actions)

    def _suggest_actions(self) -> List[str]:
        """Suggest contextually appropriate actions based on current state."""
        base_actions = [
            "Acknowledge the issue and apologize for the inconvenience.",
            "Ask for additional details or logs to investigate further.",
            "Provide a solution or workaround for the reported problem.",
            "Escalate to the appropriate team if needed.",
        ]

        # Add task-specific suggestions
        if self.task_id == TaskType.billing_refund:
            base_actions.extend([
                "Verify the invoice details and confirm the duplicate charge.",
                "Process a refund and provide transaction confirmation.",
            ])
        elif self.task_id == TaskType.csv_upload_bug:
            base_actions.extend([
                "Check CSV format and schema requirements.",
                "Request error logs or sample data for debugging.",
                "Escalate to engineering for backend investigation.",
            ])
        elif self.task_id == TaskType.sso_outage:
            base_actions.extend([
                "Verify identity provider configuration and migration details.",
                "Provide temporary access workaround if available.",
                "Escalate to security operations for SSO troubleshooting.",
            ])

        return base_actions

    def _update_resolution_state(self, action: AgentAction) -> None:
        """Update issue resolution state based on action content."""
        text = action.message.lower()

        if self.task_id == TaskType.billing_refund:
            # Resolution requires acknowledging refund and invoice reference
            if ("refund" in text or "credit" in text) and ("invoice" in text or "charge" in text):
                self.state.issue_resolved = True
        elif self.task_id == TaskType.csv_upload_bug:
            # Resolution requires escalation to engineering
            if ("escalate" in text or "engineering" in text) and ("bug" in text or "investigate" in text):
                self.state.issue_resolved = True
        elif self.task_id == TaskType.sso_outage:
            # Resolution requires SSO troubleshooting and workaround
            if ("sso" in text or "identity" in text) and ("workaround" in text or "restore" in text or "access" in text):
                self.state.issue_resolved = True

    def _check_done(self) -> bool:
        """Check if the episode should end."""
        # Episode ends if issue is resolved
        if self.state.issue_resolved:
            return True

        # Episode ends if max steps reached
        if self.state.steps_taken >= self.max_steps:
            return True

        # Episode ends if critical task and no progress after 3 steps
        if (self.task_definition.priority == "critical" and
            self.state.steps_taken >= 3 and
            len([log for log in self.logs if "[STEP]" in log]) >= 3 and
            not any("escalate" in log.lower() for log in self.state.dialogue_history)):
            return True

        return False

    def _create_penalty_step(self, reason: str) -> StepResult:
        """Create a penalty step for invalid actions."""
        penalty_reward = RewardResult(
            reward=0.01,
            graded_score=0.01,
            note=f"Penalty: {reason}",
        )

        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=penalty_reward,
            done=False,
            logs=self.logs.copy(),
        )

    def _build_reward_note(self, action: AgentAction, graded_score: float, reward_value: float) -> str:
        """Build detailed reward note for logging."""
        return (
            f"Task={self.task_id.value} steps={self.state.steps_taken} "
            f"priority={self.task_definition.priority.value} "
            f"graded={graded_score:.2f} shaped_reward={reward_value:.3f} "
            f"resolved={self.state.issue_resolved}"
        )

    def _log(self, message: str) -> None:
        """Add message to environment logs."""
        self.logs.append(message)
