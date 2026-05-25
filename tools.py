from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MOCK_ORDERS = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "status": "shipped",
        "estimated_delivery": "2026-05-05",
        "tracking_number": "TRK123456789",
        "carrier": "BlueDart",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "status": "processing",
        "estimated_delivery": "2026-05-08",
        "tracking_number": None,
        "carrier": None,
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "status": "delayed",
        "estimated_delivery": "2026-05-10",
        "tracking_number": "TRK999000111",
        "carrier": "Delhivery",
    },
}

MOCK_TICKETS: list[dict[str, Any]] = []


@dataclass
class ToolResult:
    name: str
    output: dict[str, Any]


def get_order_status(order_id: str) -> dict[str, Any]:
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {
            "found": False,
            "order_id": order_id,
            "message": "No order found for that ID.",
        }

    return {
        "found": True,
        **order,
    }


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
    "get_order_status": get_order_status,
    "create_support_ticket": create_support_ticket,
}