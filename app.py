"""
Order Support Agent

A small agentic RAG demo with:
- OpenAI provider support
- Ollama provider support
- Local ChromaDB retrieval for Ollama
- OpenAI Vector Store / file_search support for OpenAI
- Tool calling for order lookup and support-ticket creation
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

import chromadb
import ollama
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer

from tools import TOOL_REGISTRY

load_dotenv()

console = Console()


SYSTEM_PROMPT = """
You are an order support assistant for an e-commerce company.

You can:
- Answer general policy and FAQ questions using the knowledge base.
- Use get_order_status only when the user asks about a specific order and provides an order ID.
- Use create_support_ticket only when the user reports an unresolved issue such as a damaged item, delayed package, missing item, refund problem, or return problem.
- Ask for an order ID only when the user is asking about a specific order and the order ID is missing.

Critical rules:
- For general policy questions, answer directly from the knowledge base. Do not ask for an order ID.
- For "How long does shipping take?", answer using the shipping policy.
- For "Can I return an item?", answer using the return policy.
- Never invent order details.
- Never invent policy details.
- If required information is missing, ask a short clarifying question.
- Do not claim an action happened unless a tool result confirms it.
- If a tool result is present, use that result to answer the user. Do not call the same tool again.
- Keep answers concise, operational, and helpful.
""".strip()


class AgentAction(BaseModel):
    """Structured action selected by the local Ollama router."""

    action: Literal[
        "answer_directly",
        "get_order_status",
        "create_support_ticket",
        "ask_clarification",
    ] = Field(description="The next action the agent should take.")
    order_id: str | None = Field(default=None)
    issue_type: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    answer: str | None = Field(default=None)


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, str]],
        knowledge_context: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return final text plus any requested tool calls."""
        raise NotImplementedError

    @abstractmethod
    def search_knowledge_base(self, query: str) -> str:
        """Return relevant knowledge-base context for the query."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI implementation using Responses API and optional file_search."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.vector_store_id = os.getenv("VECTOR_STORE_ID")

    def _build_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "name": "get_order_status",
                "description": "Get the status and details of a customer order by order ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID, for example ORD-1001.",
                        }
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "create_support_ticket",
                "description": "Create a support ticket for an unresolved customer issue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_type": {
                            "type": "string",
                            "description": "Issue category such as damaged_item, delayed_order, missing_item, return_request, refund_issue, or other.",
                        },
                        "order_id": {
                            "type": "string",
                            "description": "The related order ID, for example ORD-1001.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short summary of the customer issue.",
                        },
                    },
                    "required": ["issue_type", "order_id", "summary"],
                    "additionalProperties": False,
                },
            },
        ]

        if self.vector_store_id:
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                }
            )

        return tools

    def _messages_to_input(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert chat-style messages to Responses API input format."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role not in {"system", "user", "assistant"}:
                role = "user"

            converted.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )

        return converted

    def generate_response(
        self,
        messages: list[dict[str, str]],
        knowledge_context: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        enhanced_messages = [dict(message) for message in messages]

        if knowledge_context:
            enhanced_messages.append(
                {
                    "role": "system",
                    "content": f"Relevant knowledge-base context:\n\n{knowledge_context}",
                }
            )

        response = self.client.responses.create(
            model=self.model,
            input=self._messages_to_input(enhanced_messages),
            tools=self._build_tools(),
        )

        tool_calls: list[dict[str, Any]] = []
        final_text_parts: list[str] = []

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "function_call":
                raw_arguments = getattr(item, "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    {
                        "name": getattr(item, "name", ""),
                        "arguments": arguments,
                        "call_id": getattr(item, "call_id", None),
                    }
                )

            elif item_type == "message":
                for content in getattr(item, "content", []):
                    text = getattr(content, "text", None)
                    if text:
                        final_text_parts.append(text)

        final_text = "\n".join(part.strip() for part in final_text_parts if part.strip())
        return final_text, tool_calls

    def search_knowledge_base(self, query: str) -> str:
        # In OpenAI mode, file_search is handled by the Responses API tool.
        return ""


class OllamaProvider(LLMProvider):
    """Ollama implementation with ChromaDB retrieval and deterministic routing."""

    def __init__(self) -> None:
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "support_docs")

        self.client = ollama.Client(host=self.base_url)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)

        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            count = self.collection.count()
            console.print(f"[green]✓ Loaded existing vector store with {count} documents[/green]")
        except Exception as exc:
            raise RuntimeError(
                "Local ChromaDB collection was not found. "
                "Run `python seed_kb.py` and choose option 2 before running the app."
            ) from exc

    def search_knowledge_base(self, query: str) -> str:
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return ""

        context_parts: list[str] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            source = metadata.get("source", "unknown") if metadata else "unknown"
            context_parts.append(
                f"Source: {source}\nDistance: {distance:.4f}\nContent:\n{document}"
            )

        return "\n\n---\n\n".join(context_parts)

    def generate_response(
        self,
        messages: list[dict[str, str]],
        knowledge_context: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Local Ollama flow.

        Instead of relying on free-form JSON hidden inside a normal assistant answer,
        this method first asks the model to choose one structured action.
        Then Python executes tools deterministically.
        """

        last_user_message = self._last_user_message(messages)

        # If a tool result already exists, skip routing and produce final answer.
        if self._has_recent_tool_result(messages):
            return self._generate_final_answer(messages, knowledge_context), []

        action = self._route_action(last_user_message, knowledge_context)

        if action.action == "get_order_status":
            if not action.order_id:
                return "Please share your order ID so I can check the status.", []

            return "", [
                {
                    "name": "get_order_status",
                    "arguments": {"order_id": action.order_id},
                }
            ]

        if action.action == "create_support_ticket":
            if not action.order_id:
                return "Please share your order ID so I can create a support ticket.", []

            issue_type = action.issue_type or "other"
            summary = action.summary or last_user_message

            return "", [
                {
                    "name": "create_support_ticket",
                    "arguments": {
                        "issue_type": issue_type,
                        "order_id": action.order_id,
                        "summary": summary,
                    },
                }
            ]

        if action.action == "ask_clarification":
            return action.answer or "Could you please share the missing details?", []

        return self._generate_final_answer(messages, knowledge_context), []

    def _route_action(
        self,
        user_message: str,
        knowledge_context: str,
    ) -> AgentAction:
        router_prompt = f"""
You are a strict router for an order-support assistant.

Return only valid JSON. Do not use markdown. Do not explain.

Allowed actions:
1. answer_directly
   Use this for general FAQ/policy questions such as shipping duration, return policy, refund policy, cancellation policy, tracking policy.

2. get_order_status
   Use this only when the user asks about a specific order and provides an order ID.

3. create_support_ticket
   Use this only when the user reports an issue that needs support action, such as damaged item, delayed order, missing item, refund issue, wrong item, or return problem.
   If the user provides an order ID, include it.
   If no order ID is provided, use ask_clarification.

4. ask_clarification
   Use this when the user asks about their own order but no order ID is provided.

Important:
- Do not ask for an order ID for general policy questions.
- "How long does shipping take?" is answer_directly.
- "Can I return an item after delivery?" is answer_directly.
- "Check order ORD-1001" is get_order_status.
- "Where is my order?" is ask_clarification.
- "I received a damaged item for order ORD-1001" is create_support_ticket.

Knowledge context, if any:
{knowledge_context or "No retrieved context."}

User message:
{user_message}

Return JSON with this schema:
{{
  "action": "answer_directly|get_order_status|create_support_ticket|ask_clarification",
  "order_id": "ORD-XXXX or null",
  "issue_type": "damaged_item|delayed_order|missing_item|refund_issue|return_request|other|null",
  "summary": "short issue summary or null",
  "answer": "clarifying question or null"
}}
""".strip()

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You return only strict JSON."},
                    {"role": "user", "content": router_prompt},
                ],
                options={"temperature": 0},
            )
        except Exception:
            return self._fallback_route(user_message)

        raw_text = response["message"]["content"]
        parsed = self._parse_json_object(raw_text)

        if parsed is None:
            return self._fallback_route(user_message)

        try:
            return AgentAction.model_validate(parsed)
        except ValidationError:
            return self._fallback_route(user_message)

    def _generate_final_answer(
        self,
        messages: list[dict[str, str]],
        knowledge_context: str = "",
    ) -> str:
        enhanced_messages = [dict(message) for message in messages]

        if knowledge_context:
            enhanced_messages.append(
                {
                    "role": "system",
                    "content": f"Relevant knowledge-base context:\n\n{knowledge_context}",
                }
            )

        enhanced_messages.append(
            {
                "role": "system",
                "content": (
                    "Answer the user's latest question. "
                    "Use the knowledge-base context and any tool result available in the conversation. "
                    "For general policy questions, do not ask for an order ID. "
                    "If a tool result says an order was found, summarize the order status clearly. "
                    "If a support ticket was created, provide the ticket ID and status. "
                    "Do not output JSON."
                ),
            }
        )

        response = self.client.chat(
            model=self.model,
            messages=enhanced_messages,
            options={"temperature": 0.2},
        )

        text = response["message"]["content"].strip()

        if self._looks_like_tool_json(text):
            return "I identified the required action, but could not produce a clean final response. Please try again."

        return text

    def _fallback_route(self, user_message: str) -> AgentAction:
        text = user_message.strip()
        lower = text.lower()

        order_id = self._extract_order_id(text)

        policy_keywords = [
            "shipping",
            "ship",
            "delivery take",
            "return policy",
            "return an item",
            "refund",
            "cancel",
            "cancellation",
            "faq",
        ]

        issue_keywords = [
            "damaged",
            "broken",
            "late",
            "delayed",
            "missing",
            "wrong item",
            "refund issue",
            "not received",
            "lost",
        ]

        order_lookup_keywords = [
            "check order",
            "order status",
            "where is",
            "track",
            "tracking",
            "my order",
        ]

        if any(keyword in lower for keyword in issue_keywords):
            if order_id:
                return AgentAction(
                    action="create_support_ticket",
                    order_id=order_id,
                    issue_type=self._infer_issue_type(lower),
                    summary=text,
                )
            return AgentAction(
                action="ask_clarification",
                answer="Please share your order ID so I can create a support ticket.",
            )

        if any(keyword in lower for keyword in policy_keywords):
            return AgentAction(action="answer_directly")

        if order_id and any(keyword in lower for keyword in order_lookup_keywords):
            return AgentAction(action="get_order_status", order_id=order_id)

        if order_id and lower.startswith(("check", "track")):
            return AgentAction(action="get_order_status", order_id=order_id)

        if "order" in lower and not order_id:
            return AgentAction(
                action="ask_clarification",
                answer="Please share your order ID so I can check the status.",
            )

        return AgentAction(action="answer_directly")

    @staticmethod
    def _infer_issue_type(lower_text: str) -> str:
        if "damaged" in lower_text or "broken" in lower_text:
            return "damaged_item"
        if "late" in lower_text or "delayed" in lower_text or "not received" in lower_text:
            return "delayed_order"
        if "missing" in lower_text:
            return "missing_item"
        if "refund" in lower_text:
            return "refund_issue"
        if "return" in lower_text:
            return "return_request"
        return "other"

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        match = re.search(r"\bORD-\d{4,}\b", text, flags=re.IGNORECASE)
        return match.group(0).upper() if match else None

    @staticmethod
    def _last_user_message(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @staticmethod
    def _has_recent_tool_result(messages: list[dict[str, str]]) -> bool:
        for message in reversed(messages[-4:]):
            if message.get("role") == "system" and message.get("content", "").startswith("Tool result:"):
                return True
        return False

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()

        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _looks_like_tool_json(text: str) -> bool:
        parsed = OllamaProvider._parse_json_object(text)
        return isinstance(parsed, dict) and ("tool" in parsed or "action" in parsed)


class OrderSupportAgent:
    """Main agent orchestrator."""

    def __init__(self) -> None:
        provider_name = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        if provider_name == "openai":
            console.print("[bold blue]Using OpenAI[/bold blue]")
            self.provider: LLMProvider = OpenAIProvider()
        elif provider_name == "ollama":
            console.print("[bold green]Using Ollama (Local LLM)[/bold green]")
            self.provider = OllamaProvider()
            console.print(
                f"[green]✓ Using Ollama with model: {os.getenv('OLLAMA_MODEL', 'llama3.2:latest')}[/green]"
            )
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER={provider_name!r}. Use 'openai' or 'ollama'."
            )

        self.conversation_history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def ask(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})

        max_iterations = 5

        for _ in range(max_iterations):
            knowledge_context = self.provider.search_knowledge_base(user_input)

            response_text, tool_calls = self.provider.generate_response(
                self.conversation_history,
                knowledge_context=knowledge_context,
            )

            if not tool_calls:
                final_response = response_text.strip() or (
                    "I could not generate a response. Please try again."
                )
                self.conversation_history.append(
                    {"role": "assistant", "content": final_response}
                )
                return final_response

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})

                if tool_name not in TOOL_REGISTRY:
                    tool_result = {
                        "error": f"Unknown tool: {tool_name}",
                    }
                else:
                    try:
                        tool_result = TOOL_REGISTRY[tool_name](**arguments)
                    except Exception as exc:
                        tool_result = {
                            "error": str(exc),
                            "tool": tool_name,
                            "arguments": arguments,
                        }

                self.conversation_history.append(
                    {
                        "role": "system",
                        "content": (
                            "Tool result: "
                            + json.dumps(
                                {
                                    "tool": tool_name,
                                    "arguments": arguments,
                                    "result": tool_result,
                                },
                                ensure_ascii=False,
                            )
                        ),
                    }
                )

                self.conversation_history.append(
                    {
                        "role": "system",
                        "content": (
                            "The tool has already been executed. "
                            "Now answer the user using the tool result above. "
                            "Do not call the same tool again."
                        ),
                    }
                )

        fallback = "I apologize, but I could not complete the request after multiple attempts."
        self.conversation_history.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self) -> None:
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    try:
        agent = OrderSupportAgent()
    except Exception as exc:
        console.print(f"[bold red]Startup failed:[/bold red] {exc}")
        raise

    console.print(
        Panel.fit(
            "[bold cyan]Order Support Agent v2[/bold cyan]\n"
            "OpenAI + Ollama + RAG + Tool Calling\n"
            "Type [bold]exit[/bold], [bold]quit[/bold], or [bold]q[/bold] to end.\n"
            "Type [bold]reset[/bold] to clear conversation history.",
            border_style="cyan",
        )
    )

    while True:
        user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "reset":
            agent.reset()
            console.print("[yellow]Conversation history reset.[/yellow]")
            continue

        if not user_input:
            continue

        try:
            response = agent.ask(user_input)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            break
        except Exception as exc:
            response = f"Runtime error: {exc}"

        console.print(f"\n[bold blue]Agent:[/bold blue] {response}")


if __name__ == "__main__":
    main()
