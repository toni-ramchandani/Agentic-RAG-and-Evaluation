from __future__ import annotations

import json
import os
from typing import Any
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from tools import TOOL_REGISTRY

console = Console()

SYSTEM_PROMPT = """
You are an order support assistant.

You can:
- Answer policy and FAQ questions using the provided file search tool
- Ask for an order ID when the user asks about a specific order and the ID is missing
- Use get_order_status for order lookups
- Use create_support_ticket when the issue cannot be resolved directly

Rules:
- Never invent order details or policy details
- If required information is missing, ask a short clarifying question
- Do not claim an action happened unless a tool result confirms it
- Keep answers concise, helpful, and operational
- End with a short summary of what you did
""".strip()


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: list[dict], knowledge_context: str = "") -> tuple[str, list[dict]]:
        """Generate a response and return (text, tool_calls)"""
        pass
    
    @abstractmethod
    def search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base and return relevant context"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.vector_store_id = os.getenv("VECTOR_STORE_ID", "").strip()
        
        if not self.vector_store_id:
            console.print("[yellow]Warning: VECTOR_STORE_ID not set. Run seed_kb.py first.[/yellow]")
    
    def generate_response(self, messages: list[dict], knowledge_context: str = "") -> tuple[str, list[dict]]:
        """Generate response using OpenAI Responses API"""
        tools = self._get_tools()
        
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            tools=tools,
            include=["file_search_call.results"] if self.vector_store_id else [],
        )
        
        # Extract function calls
        function_calls = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                function_calls.append({
                    "name": item.name,
                    "arguments": json.loads(item.arguments),
                    "call_id": item.call_id
                })
        
        return response.output_text, function_calls
    
    def search_knowledge_base(self, query: str) -> str:
        """OpenAI uses built-in file_search, so this returns empty"""
        return ""
    
    def _get_tools(self) -> list[dict]:
        tools = []
        
        # Add file search if vector store exists
        if self.vector_store_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
                "max_num_results": 3,
            })
        
        # Add function tools
        tools.extend([
            {
                "type": "function",
                "name": "get_order_status",
                "description": "Look up a customer's order by order ID.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID, for example ORD-1001",
                        }
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "create_support_ticket",
                "description": "Create a support ticket for an unresolved order issue.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_type": {
                            "type": "string",
                            "description": "A short issue category such as delayed_order or damaged_item",
                        },
                        "order_id": {
                            "type": "string",
                            "description": "The order ID, for example ORD-1001",
                        },
                        "summary": {
                            "type": "string",
                            "description": "A concise description of the problem",
                        },
                    },
                    "required": ["issue_type", "order_id", "summary"],
                    "additionalProperties": False,
                },
            },
        ])
        
        return tools


class OllamaProvider(LLMProvider):
    """Ollama (local LLM) provider"""
    
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        
        # Initialize local vector store
        self._init_vector_store()
        
        console.print(f"[green]✓[/green] Using Ollama with model: {self.model}")
    
    def _init_vector_store(self):
        """Initialize ChromaDB for local vector search"""
        use_local = os.getenv("USE_LOCAL_VECTOR_STORE", "true").lower() == "true"
        
        if use_local:
            import chromadb
            from chromadb.utils import embedding_functions
            
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            # Use sentence transformers for embeddings
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name="support_docs",
                    embedding_function=self.embedding_function
                )
                console.print(f"[green]✓[/green] Loaded existing vector store with {self.collection.count()} documents")
            except:
                self.collection = None
                console.print("[yellow]Warning: No vector store found. Run seed_kb.py first.[/yellow]")
        else:
            self.collection = None
    
    def generate_response(self, messages: list[dict], knowledge_context: str = "") -> tuple[str, list[dict]]:
        """Generate response using Ollama"""
        
        # Prepare messages with context
        enhanced_messages = list(messages)
        if knowledge_context:
            enhanced_messages.append({
                "role": "system",
                "content": f"Relevant information from knowledge base:\n\n{knowledge_context}"
            })
        
        # Add tool descriptions to the system prompt
        tool_prompt = self._get_tool_prompt()
        enhanced_messages[0]["content"] += f"\n\n{tool_prompt}"
        
        # Generate response
        response = self.client.chat(
            model=self.model,
            messages=enhanced_messages,
        )
        
        response_text = response["message"]["content"]
        
        # Parse for tool calls (simple JSON detection)
        tool_calls = self._extract_tool_calls(response_text)
        
        # Remove tool call JSON from response text if found
        if tool_calls:
            for call in tool_calls:
                response_text = response_text.replace(call.get("_raw", ""), "").strip()
        
        return response_text, tool_calls
    
    def search_knowledge_base(self, query: str) -> str:
        """Search ChromaDB for relevant context"""
        if not self.collection:
            return ""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if results and results["documents"]:
            context_parts = []
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                source = metadata.get("source", "unknown")
                context_parts.append(f"From {source}:\n{doc}")
            return "\n\n".join(context_parts)
        
        return ""
    
    def _get_tool_prompt(self) -> str:
        """Generate tool descriptions for the LLM"""
        return """
Available Tools (respond with JSON only when calling a tool):

1. get_order_status - Look up order information
   Format: {"tool": "get_order_status", "order_id": "ORD-XXXX"}

2. create_support_ticket - Create a support ticket
   Format: {"tool": "create_support_ticket", "issue_type": "type", "order_id": "ORD-XXXX", "summary": "description"}

When you need to call a tool, respond ONLY with the JSON object. Otherwise, respond normally.
"""
    
    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        # Look for JSON objects in the response
        import re
        json_pattern = r'\{[^}]*"tool"[^}]*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                call_data = json.loads(match.group())
                if "tool" in call_data:
                    tool_name = call_data.pop("tool")
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": call_data,
                        "call_id": f"call_{len(tool_calls)}",
                        "_raw": match.group()
                    })
            except json.JSONDecodeError:
                continue
        
        return tool_calls


class OrderSupportAgent:
    def __init__(self) -> None:
        load_dotenv()
        
        # Determine which provider to use
        provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if provider_name == "openai":
            console.print("[cyan]Using OpenAI API[/cyan]")
            self.provider = OpenAIProvider()
        else:
            console.print("[cyan]Using Ollama (Local LLM)[/cyan]")
            self.provider = OllamaProvider()
        
        self.conversation_history: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
    
    def ask(self, user_text: str) -> str:
        """Process user query and return response"""
        
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        # Search knowledge base (for Ollama)
        knowledge_context = self.provider.search_knowledge_base(user_text)
        
        # Generate response
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            response_text, tool_calls = self.provider.generate_response(
                self.conversation_history,
                knowledge_context
            )
            
            if not tool_calls:
                # No tool calls, we're done
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                return response_text
            
            # Execute tool calls
            for call in tool_calls:
                tool_name = call["name"]
                arguments = call["arguments"]
                
                if tool_name in TOOL_REGISTRY:
                    result = TOOL_REGISTRY[tool_name](**arguments)
                    
                    # Add tool result to conversation
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"Tool {tool_name} result: {json.dumps(result)}"
                    })
            
            iteration += 1
        
        # If we hit max iterations, return  what we have
        return response_text or "I apologize, but I'm having trouble processing your request. Please try again."


def main() -> None:
    """Main conversation loop"""
    try:
        agent = OrderSupportAgent()
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    console.print("\n[bold cyan]Order Support Agent v2[/bold cyan]")
    console.print("[dim]Now with OpenAI and Ollama support![/dim]")
    console.print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        try:
            user_text = console.input("[bold green]You:[/bold green] ").strip()
            
            if user_text.lower() in {"exit", "quit", "q"}:
                console.print("\n[cyan]Thank you for using Order Support Agent. Goodbye![/cyan]")
                break
            
            if not user_text:
                continue
            
            answer = agent.ask(user_text)
            console.print(f"\n[bold blue]Agent:[/bold blue] {answer}\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()        