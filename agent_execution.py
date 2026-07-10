from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TRACES_DIR = Path("traces")


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ExecutionStep:
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: str | None = None
    started_at_ms: int = field(default_factory=now_ms)
    ended_at_ms: int | None = None
    latency_ms: int | None = None

    def finish(
        self,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        self.ended_at_ms = now_ms()
        self.latency_ms = self.ended_at_ms - self.started_at_ms
        self.status = status
        self.error = error

        if output:
            self.output.update(output)

        if metadata:
            self.metadata.update(metadata)


@dataclass
class AgentExecution:
    user_input: str
    provider: str
    model: str | None = None
    session_id: str | None = None
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    started_at_ms: int = field(default_factory=now_ms)
    ended_at_ms: int | None = None
    total_latency_ms: int | None = None
    status: str = "running"
    final_answer: str | None = None
    error: str | None = None
    steps: list[ExecutionStep] = field(default_factory=list)

    def start_step(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionStep:
        step = ExecutionStep(
            name=name,
            input=input or {},
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def finish(
        self,
        final_answer: str | None = None,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        self.ended_at_ms = now_ms()
        self.total_latency_ms = self.ended_at_ms - self.started_at_ms
        self.status = status
        self.final_answer = final_answer
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "user_input": self.user_input,
            "status": self.status,
            "error": self.error,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "total_latency_ms": self.total_latency_ms,
            "final_answer": self.final_answer,
            "steps": [asdict(step) for step in self.steps],
        }

    def save(self, traces_dir: Path = TRACES_DIR) -> Path:
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.trace_id}.json"

        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return path
