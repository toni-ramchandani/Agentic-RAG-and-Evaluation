# Agentic RAG and Evaluation

This is a demonstration of an agentic RAG (Retrieval Augmented Generation) system that uses local LLMs to answer questions about an e-commerce order support system.

## Features

- Uses local LLMs (Ollama) for question answering
- Retrieves relevant information from knowledge base documents
- Supports multiple knowledge base types (ChromaDB, OpenAI Vector Store, or both)
- Provides a simple command-line interface for interaction

## Setup

1. Install Ollama (https://ollama.com/download)
2. Pull a local LLM model:
   ```
   ollama pull llama3
   ```
3. Install dependencies:
   ```
   pip install openai ollama python-dotenv pydantic chromadb sentence-transformers rich
   ```

## Usage

Run the application:
```
python app.py
```

## Knowledge Base

The system uses markdown files in the `data/` directory as the knowledge base:
- `faq.md` - Frequently Asked Questions
- `shipping-policy.md` - Shipping policies
- `returns-policy.md` - Returns policies

## Files

- `app.py` - Main application entry point
- `tools.py` - Tools for interacting with the LLM and knowledge base
- `seed_kb.py` - Script to seed the knowledge base with documents
- `requirements.txt` - Dependencies
- `data/` - Directory containing knowledge base documents