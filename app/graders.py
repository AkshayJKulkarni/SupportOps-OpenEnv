"""Grading logic for support actions in SupportOps OpenEnv."""

from .models import AgentAction, TaskType


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1)."""
    return round(max(0.01, min(score, 0.99)), 3)


def grade_action(task_id: TaskType, action: AgentAction) -> float:
    """Deterministic grading returns a score strictly between 0.0 and 1.0."""
    message = action.message.strip().lower()
    if task_id == TaskType.billing_refund:
        return _clamp(_grade_billing_refund(message))
    if task_id == TaskType.csv_upload_bug:
        return _clamp(_grade_csv_upload_bug(message))
    if task_id == TaskType.sso_outage:
        return _clamp(_grade_sso_outage(message))
    return 0.01


def _grade_billing_refund(message: str) -> float:
    score = 0.0
    if "refund" in message:
        score += 0.35
    if "invoice" in message or "charge" in message:
        score += 0.25
    if "pro" in message or "subscription" in message:
        score += 0.15
    if "apolog" in message or "sorry" in message:
        score += 0.15
    if "resolve" in message or "process" in message:
        score += 0.1
    return score


def _grade_csv_upload_bug(message: str) -> float:
    score = 0.0
    if "csv" in message and "upload" in message:
        score += 0.3
    if "error" in message or "rows" in message or "import" in message:
        score += 0.2
    if "log" in message or "schema" in message or "format" in message:
        score += 0.2
    if "investigate" in message or "escalate" in message or "engineering" in message:
        score += 0.2
    if "follow up" in message or "update" in message:
        score += 0.1
    return score


def _grade_sso_outage(message: str) -> float:
    score = 0.0
    if "sso" in message or "single sign" in message or "idp" in message:
        score += 0.3
    if "403" in message or "forbidden" in message or "outage" in message:
        score += 0.25
    if "okta" in message or "identity provider" in message:
        score += 0.2
    if "escalate" in message or "security" in message or "support" in message:
        score += 0.15
    if "investigate" in message or "restore" in message or "workaround" in message:
        score += 0.1
    return score
