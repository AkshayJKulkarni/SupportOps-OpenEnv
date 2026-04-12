"""Pydantic models for the SupportOps OpenEnv environment."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    billing_refund = "billing_refund"
    csv_upload_bug = "csv_upload_bug"
    sso_outage = "sso_outage"


class ActionType(str, Enum):
    response = "response"
    escalation = "escalation"
    request_information = "request_information"
    apology = "apology"
    workaround = "workaround"


class SerializationMixin:
    """Mixin for consistent JSON serialization and deserialization."""

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "SerializationMixin":
        return cls.model_validate_json(payload)


class Ticket(BaseModel, SerializationMixin):
    task_id: TaskType
    title: str
    description: str
    customer_name: str
    account_tier: str
    created_at: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class TicketAction(BaseModel, SerializationMixin):
    message: str
    action_type: ActionType = ActionType.response
    step_hint: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class TicketState(BaseModel, SerializationMixin):
    ticket: Ticket
    dialogue_history: List[str] = Field(default_factory=list)
    issue_resolved: bool = False
    escalation_required: bool = False
    steps_taken: int = 0
    reward_score: float = 0.01
    done: bool = False


class TicketObservation(BaseModel, SerializationMixin):
    state: TicketState
    available_actions: List[str] = Field(default_factory=list)

    def available_action_types(self) -> List[str]:
        return [action.value for action in ActionType]


class RewardModel(BaseModel, SerializationMixin):
    reward: float
    graded_score: float
    note: str
    details: Optional[Dict[str, Any]] = None


class StepResult(BaseModel, SerializationMixin):
    observation: TicketObservation
    reward: RewardModel
    done: bool
    logs: List[str]


class ResetResult(BaseModel, SerializationMixin):
    observation: TicketObservation
    logs: List[str]


class OpenAIRequest(BaseModel, SerializationMixin):
    prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512


# Aliases for backwards compatibility
Observation = TicketObservation
EnvState = TicketState
RewardResult = RewardModel
AgentAction = TicketAction


# API Response Models
class HealthResponse(BaseModel, SerializationMixin):
    status: str
    message: str
    version: str = "0.1.0"
    timestamp: str


class ResetRequest(BaseModel, SerializationMixin):
    task_id: Optional[TaskType] = None


class ResetResponse(BaseModel, SerializationMixin):
    observation: TicketObservation
    logs: List[str]


class StepResponse(BaseModel, SerializationMixin):
    observation: TicketObservation
    reward: RewardModel
    done: bool
    logs: List[str]


class StateResponse(BaseModel, SerializationMixin):
    observation: TicketObservation
    logs: List[str]


class ErrorResponse(BaseModel, SerializationMixin):
    error: str
    detail: Optional[str] = None
    code: int
