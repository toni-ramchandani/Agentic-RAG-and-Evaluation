from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()
DATA_DIR = Path("data")


def extract_order_id(text: str) -> str | None:
    match = re.search(r"\bORD-\d{4,}\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def build_metadata(path: Path, chunk: str, chunk_index: int) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "source": path.name,
        "chunk": chunk_index,
    }

    if path.name.lower() == "order.md":
        metadata["document_type"] = "order"

        order_id = extract_order_id(chunk)
        if order_id:
            metadata["order_id"] = order_id

    return metadata


def seed_openai_vector_store():
    """Upload files to OpenAI vector store"""
    from openai import OpenAI

    console.print("\n[cyan]Setting up OpenAI Vector Store...[/cyan]")
    client = OpenAI()

    vector_store_name = "Order Support KB"
    vector_store = client.beta.vector_stores.create(name=vector_store_name)

    console.print(f"[green]✓[/green] Created vector store: {vector_store.id}")

    for path in DATA_DIR.glob("*.md"):
        with path.open("rb") as f:
            client.beta.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store.id,
                file=f,
            )
        console.print(f"[green]✓[/green] Uploaded: {path.name}")

    console.print("\n[bold yellow]Add this to your .env file:[/bold yellow]")
    console.print(f"[yellow]VECTOR_STORE_ID={vector_store.id}[/yellow]")
    console.print("[yellow]LLM_PROVIDER=openai[/yellow]\n")


def seed_chroma_vector_store():
    """Create local ChromaDB vector store"""
    import chromadb
    from chromadb.utils import embedding_functions

    console.print("\n[cyan]Setting up Local ChromaDB Vector Store...[/cyan]")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    chroma_client = chromadb.PersistentClient(path=persist_dir)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        chroma_client.delete_collection(name="support_docs")
        console.print("[yellow]→[/yellow] Deleted existing collection")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="support_docs",
        embedding_function=embedding_function,
        metadata={"description": "Order support knowledge base"},
    )

    console.print(f"[green]✓[/green] Created ChromaDB collection at: {persist_dir}")

    doc_count = 0

    for path in DATA_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8")

        # Existing behavior preserved: paragraph-based chunks.
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        for i, chunk in enumerate(chunks):
            metadata = build_metadata(path, chunk, i)

            collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[f"{path.stem}_{i}"],
            )
            doc_count += 1

        console.print(f"[green]✓[/green] Indexed: {path.name} ({len(chunks)} chunks)")

    console.print(f"\n[bold green]Success![/bold green] Indexed {doc_count} document chunks")
    console.print("\n[bold yellow]Ensure these settings are in your .env file:[/bold yellow]")
    console.print("[yellow]USE_LOCAL_VECTOR_STORE=true[/yellow]")
    console.print(f"[yellow]CHROMA_PERSIST_DIR={persist_dir}[/yellow]")
    console.print("[yellow]LLM_PROVIDER=ollama[/yellow]\n")


def main() -> None:
    load_dotenv()

    console.print("[bold cyan]Order Support Agent - Knowledge Base Seeder[/bold cyan]")
    console.print("=" * 60)

    if not DATA_DIR.exists():
        console.print(f"[red]Error: {DATA_DIR} directory not found![/red]")
        return

    md_files = list(DATA_DIR.glob("*.md"))
    if not md_files:
        console.print(f"[red]Error: No .md files found in {DATA_DIR}[/red]")
        return

    console.print(f"Found {len(md_files)} markdown files to index")

    console.print("\n[bold]Select knowledge base type:[/bold]")
    console.print("  [cyan]1.[/cyan] OpenAI Vector Store (cloud, requires API key)")
    console.print("  [cyan]2.[/cyan] ChromaDB (local, free, works with Ollama)")
    console.print("  [cyan]3.[/cyan] Both")

    choice = console.input("\nEnter choice [bold](1/2/3)[/bold]: ").strip()

    try:
        if choice == "1":
            seed_openai_vector_store()
        elif choice == "2":
            seed_chroma_vector_store()
        elif choice == "3":
            seed_openai_vector_store()
            seed_chroma_vector_store()
        else:
            console.print("[red]Invalid choice. Exiting.[/red]")
            return

        console.print("[bold green]✓ Knowledge base setup complete![/bold green]")
        console.print("You can now run: [bold cyan]python app.py[/bold cyan]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\nFor ChromaDB, make sure dependencies are installed:")
        console.print("  [cyan]pip install chromadb sentence-transformers[/cyan]")


if __name__ == "__main__":
    main()
