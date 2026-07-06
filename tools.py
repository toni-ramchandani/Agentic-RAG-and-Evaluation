from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MOCK_TICKETS: list[dict[str, Any]] = []


@dataclass
class ToolResult:
    name: str
    output: dict[str, Any]


def create_support_ticket(issue_type: str, order_id: str, summary: str) -> dict[str, Any]:
    ticket_id = f"TICKET-{len(MOCK_TICKETS) + 1:04d}"
    ticket = {
        "ticket_id": ticket_id,
        "issue_type": issue_type,
        "order_id": order_id,
        "summary": summary,
        "status": "open",
    }
    MOCK_TICKETS.append(ticket)
    return ticket


TOOL_REGISTRY = {
    "create_support_ticket": create_support_ticket,
}
