"""Task definitions and scenario templates for the SupportOps OpenEnv environment."""

from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, Field
from .models import TaskType, Ticket


class TaskPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class TaskDefinition(BaseModel):
    ticket: Ticket
    workflow_steps: List[str] = Field(default_factory=list)
    grading_metadata: Dict[str, str] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.medium
    context: str = ""


TASK_DEFINITIONS: Dict[TaskType, TaskDefinition] = {
    TaskType.billing_refund: TaskDefinition(
        ticket=Ticket(
            task_id=TaskType.billing_refund,
            title="Refund request for an overcharge on annual renewal",
            description=(
                "Eva Martinez reports that her Pro plan was charged twice during the annual renewal cycle. "
                "She needs the duplicated charge reversed and asks for confirmation that the refund processed."
            ),
            customer_name="Eva Martinez",
            account_tier="Pro",
            created_at="2026-04-08T09:32:00Z",
            metadata={
                "invoice_id": "INV-2026-0041",
                "amount": "$199.99",
                "payment_method": "credit_card",
                "renewal_type": "annual",
                "customer_segment": "smarthome",
            },
        ),
        workflow_steps=[
            "Acknowledge the invoice issue and apologize for the inconvenience.",
            "Confirm the duplicate charge and verify the invoice details.",
            "Process a refund or credit and document the transaction.",
            "Communicate the expected timeline and follow up with next steps.",
            "Escalate to billing operations if there is any dispute over the payment method.",
        ],
        grading_metadata={
            "expected_keywords": "refund, invoice, duplicate charge, apologize, confirm",
            "requires_refund": "true",
            "needs_follow_up": "true",
            "customer_type": "Pro",
        },
        priority=TaskPriority.high,
        context=(
            "The customer is on a Pro SaaS plan and depends on the platform for monthly reporting. "
            "A billing error during an annual renewal can damage trust, so the support response should be prompt, transparent, and action-oriented."
        ),
    ),
    TaskType.csv_upload_bug: TaskDefinition(
        ticket=Ticket(
            task_id=TaskType.csv_upload_bug,
            title="CSV import failure blocking analytics deployment",
            description=(
                "Jordan Lee from a Business account cannot import their analytics dataset. "
                "The CSV upload interface returns a generic failure and the project import stalls, threatening a customer-facing rollout."
            ),
            customer_name="Jordan Lee",
            account_tier="Business",
            created_at="2026-04-08T09:15:00Z",
            metadata={
                "workspace_id": "ws-analytic-37",
                "csv_size_mb": "12.4",
                "affected_module": "Analytics Import",
                "rows": "4,812",
                "file_type": "CSV",
            },
        ),
        workflow_steps=[
            "Acknowledge the upload failure and ask for the exact error details.",
            "Validate the CSV format, field schema, and row/column expectations.",
            "Check for permission, workspace limits, or known platform bug patterns.",
            "Escalate to engineering with logs if the issue appears to be a backend bug.",
            "Provide an interim workaround or manual import path for the rollout."
        ],
        grading_metadata={
            "expected_keywords": "csv upload, error, schema, logs, escalate, workaround",
            "bug_escalation": "required",
            "has_analytics_impact": "true",
            "business_customer": "true",
        },
        priority=TaskPriority.critical,
        context=(
            "A Business-tier customer is attempting to onboard an analytics dataset before a scheduled release. "
            "The problem is likely in SaaS import validation or file handling, so the support path should combine customer empathy with technical troubleshooting."
        ),
    ),
    TaskType.sso_outage: TaskDefinition(
        ticket=Ticket(
            task_id=TaskType.sso_outage,
            title="Enterprise SSO outage after identity provider migration",
            description=(
                "Aria Patel reports that Helix Dynamics users cannot authenticate through SSO after migrating to Okta. "
                "All workspace login attempts fail with 403 errors, and the security team is concerned about access loss."
            ),
            customer_name="Aria Patel",
            account_tier="Enterprise",
            created_at="2026-04-08T09:02:00Z",
            metadata={
                "company_name": "Helix Dynamics",
                "idp_provider": "Okta",
                "error_code": "403",
                "affected_users": "120",
                "sso_channel": "Okta SAML",
            },
        ),
        workflow_steps=[
            "Confirm the outage, log the SSO error code, and acknowledge the business impact.",
            "Verify the identity provider configuration and recent migration details.",
            "Recommend a temporary access workaround if available.",
            "Escalate to security or SSO operations if the provider integration appears broken.",
            "Communicate the recovery plan and expected time to restore access."
        ],
        grading_metadata={
            "expected_keywords": "SSO, IdP, 403, migration, Okta, outage, access restore",
            "must_offer_workaround": "true",
            "escalation_path": "security/SSO operations",
            "enterprise_impact": "high",
        },
        priority=TaskPriority.critical,
        context=(
            "This is an Enterprise SaaS customer experiencing a service-critical login outage. "
            "The support agent must treat the request as urgent, collect migration context, and coordinate with identity provider troubleshooting."
        ),
    ),
}


DEFAULT_TASKS: Dict[TaskType, Ticket] = {
    task_id: task_definition.ticket
    for task_id, task_definition in TASK_DEFINITIONS.items()
}


def list_task_labels() -> List[str]:
    return [task.value for task in TASK_DEFINITIONS.keys()]
