# 🤖 Order Support Agent v2

**An AI-powered customer support chatbot that runs on both OpenAI and local LLMs (Ollama)**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Architecture](#-architecture)
4. [Installation](#-installation)
5. [Quick Start](#-quick-start)
6. [Configuration](#-configuration)
7. [Usage](#-usage)
8. [Project Structure](#-project-structure)
9. [LLM Providers](#-llm-providers)
10. [Knowledge Base](#-knowledge-base)
11. [Tools & Functions](#-tools--functions)
12. [Advanced Usage](#-advanced-usage)
13. [Troubleshooting](#-troubleshooting)
14. [Performance & Costs](#-performance--costs)
15. [Development](#-development)
16. [Production Deployment](#-production-deployment)
17. [Contributing](#-contributing)
18. [License](#-license)

---

## 🎯 Overview

The **Order Support Agent** is a conversational AI assistant designed to handle e-commerce customer support queries. It combines:

- **RAG (Retrieval-Augmented Generation)**: Searches through policy documents to provide accurate answers
- **Function Calling**: Executes tools for order lookups and ticket creation
- **Multi-Provider Support**: Works with both OpenAI's GPT models and local LLMs via Ollama
- **Rich CLI**: Beautiful terminal interface with colored output

### What It Can Do

✅ Answer customer questions about:
- Shipping policies and delivery times
- Return and refund processes
- Frequently asked questions
- Payment methods and security

✅ Perform actions:
- Look up order status by ID
- Create support tickets for issues
- Ask clarifying questions when needed

✅ Run anywhere:
- **Cloud**: Use OpenAI's powerful GPT-4 models
- **Local**: Run completely offline with Ollama

---

## ✨ Features

### Core Capabilities
- 🗣️ **Natural Conversations**: Maintains context across multiple turns
- 📚 **Knowledge Base Integration**: Searches policy documents for accurate information
- 🔧 **Tool Execution**: Calls functions to perform actions (order lookup, ticket creation)
- 🎨 **Beautiful CLI**: Rich terminal UI with colors and formatting
- 🔄 **Dual Provider Support**: Switch between OpenAI and Ollama seamlessly

### Technical Features
- **Abstract Provider Pattern**: Easy to add new LLM providers
- **Local Vector Database**: ChromaDB for offline document search
- **Conversation History**: Tracks full conversation context
- **Error Handling**: Graceful degradation and helpful error messages
- **Configuration Management**: Environment-based settings
- **Type Safety**: Full type hints with Pydantic

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INPUT                          │
│                   "Check order ORD-1001"                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  OrderSupportAgent                          │
│  • Manages conversation history                            │
│  • Routes to appropriate LLM provider                       │
│  • Orchestrates tool calls                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ OpenAI Provider │       │ Ollama Provider │
│  • GPT-4.1      │       │  • Llama 3.2    │
│  • Cloud API    │       │  • Local only   │
│  • Vector Store │       │  • ChromaDB     │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Knowledge Base       │
         │  • faq.md              │
         │  • shipping_policy.md  │
         │  • returns_policy.md   │
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Tool Registry        │
         │  • get_order_status    │
         │  • create_ticket       │
         └────────────────────────┘
```

### Component Breakdown

| Component | Purpose | File |
|-----------|---------|------|
| **OrderSupportAgent** | Main orchestrator, manages conversation | `app.py` |
| **LLMProvider (Abstract)** | Interface for LLM providers | `app.py` |
| **OpenAIProvider** | OpenAI GPT integration | `app.py` |
| **OllamaProvider** | Local LLM via Ollama | `app.py` |
| **TOOL_REGISTRY** | Maps tool names to functions | `tools.py` |
| **Knowledge Base** | Policy documents in Markdown | `data/` |
| **Vector Store Seeder** | Indexes documents for search | `seed_kb.py` |

---

## 📦 Installation

### Prerequisites

- **Python 3.10+** (3.13 recommended)
- **pip** or **uv** package manager
- **Git** (for cloning)

### Option 1: OpenAI (Cloud)
```bash
# Just Python and an API key
pip install -r requirements.txt
```

### Option 2: Ollama (Local)
```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Mac: brew install ollama
# Linux: curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Install Python dependencies
pip install -r requirements.txt
```

### Step-by-Step Setup

```bash
# 1. Clone or navigate to the project
cd evalExperiments

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Mac/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env with your settings

# 6. Seed knowledge base
python seed_kb.py

# 7. Run the agent
python app.py
```

---

## 🚀 Quick Start

### Using Ollama (Local, Free)

```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model
ollama pull llama3.2

# 3. Configure .env
echo "LLM_PROVIDER=ollama" > .env
echo "OLLAMA_MODEL=llama3.2:latest" >> .env
echo "USE_LOCAL_VECTOR_STORE=true" >> .env

# 4. Seed knowledge base
python seed_kb.py
# Choose option 2 (ChromaDB)

# 5. Run
python app.py
```

### Using OpenAI (Cloud, Paid)

```bash
# 1. Get API key from https://platform.openai.com/api-keys

# 2. Configure .env
echo "LLM_PROVIDER=openai" > .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
echo "OPENAI_MODEL=gpt-4.1" >> .env

# 3. Seed knowledge base
python seed_kb.py
# Choose option 1 (OpenAI)

# 4. Run
python app.py
```

### First Conversation

```
You: What's your shipping policy?

Agent: We offer three shipping options:
- Standard: 5-7 business days ($5.99, free over $50)
- Express: 2-3 business days ($12.99)
- Overnight: 1 business day ($24.99)

You: Check order ORD-1001

Agent: Your order ORD-1001 has shipped via BlueDart.
Tracking: TRK123456789
Estimated delivery: May 5, 2026

You: exit
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# ============================================
# LLM Provider Selection
# ============================================
# Options: "openai" or "ollama"
LLM_PROVIDER=ollama

# ============================================
# OpenAI Settings (if using OpenAI)
# ============================================
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_MODEL=gpt-4.1
# Vector store ID from seed_kb.py
VECTOR_STORE_ID=vs_xxxxx

# ============================================
# Ollama Settings (if using Ollama)
# ============================================
OLLAMA_BASE_URL=http://localhost:11434
# Available models: llama3.2, mistral, codellama, etc.
OLLAMA_MODEL=llama3.2:latest

# ============================================
# Knowledge Base Settings
# ============================================
# Use local ChromaDB (required for Ollama)
USE_LOCAL_VECTOR_STORE=true
# Where to store the vector database
CHROMA_PERSIST_DIR=./chroma_db
```

### LLM Model Options

#### OpenAI Models
| Model | Best For | Cost/1M Tokens | Speed |
|-------|----------|----------------|-------|
| `gpt-4.1` | Production quality | $15/$60 | Medium |
| `gpt-4-turbo-preview` | Fast responses | $10/$30 | Fast |
| `gpt-3.5-turbo` | Budget-friendly | $0.50/$1.50 | Very Fast |

#### Ollama Models
| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| `llama3.2:latest` | 8B | General purpose | Fast |
| `llama3.2:1b` | 1B | Very fast responses | Very Fast |
| `mistral:latest` | 7B | Instruction following | Fast |
| `qwen2.5:7b` | 7B | Multilingual | Fast |

---

## 🎮 Usage

### Basic Commands

```bash
# Start the agent
python app.py

# Seed/reseed knowledge base
python seed_kb.py

# Run tests (if available)
python -m pytest

# Check Python environment
python --version
```

### Example Queries

#### Knowledge Base Questions
```
You: How do I return an item?
You: What payment methods do you accept?
You: How long does shipping take?
You: Do you ship internationally?
```

#### Order Operations
```
You: Check order ORD-1001
You: What's the status of my order ORD-1002?
You: Track order ORD-1003
```

#### Ticket Creation
```
You: I received a damaged item for order ORD-1001
You: My order is late, it's ORD-1002
You: I want to return order ORD-1001
```

#### Multi-Turn Conversations
```
You: I have a question about shipping
Agent: I'd be happy to help! What would you like to know?
You: How much does express shipping cost?
Agent: Express shipping costs $12.99 and delivers in 2-3 business days.
You: And what about overnight?
Agent: Overnight shipping costs $24.99 for next business day delivery.
```

---

## 📁 Project Structure

```
evalExperiments/
│
├── 📄 .env                      # Your configuration (gitignored)
├── 📄 .env.example              # Configuration template
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
│
├── 📄 app.py                    # Main application
│   ├── OrderSupportAgent        # Main orchestrator
│   ├── LLMProvider (Abstract)   # Provider interface
│   ├── OpenAIProvider           # OpenAI implementation
│   └── OllamaProvider           # Ollama implementation
│
├── 📄 tools.py                  # Tool definitions
│   ├── MOCK_ORDERS             # Sample order data
│   ├── get_order_status()      # Order lookup function
│   ├── create_support_ticket() # Ticket creation function
│   └── TOOL_REGISTRY           # Function registry
│
├── 📄 seed_kb.py               # Knowledge base seeder
│   ├── seed_openai_vector_store()   # OpenAI indexing
│   └── seed_chroma_vector_store()   # ChromaDB indexing
│
├── 📁 data/                     # Knowledge base documents
│   ├── faq.md                  # Common questions
│   ├── shipping_policy.md      # Shipping information
│   └── returns_policy.md       # Returns & refunds
│
├── 📁 chroma_db/               # Local vector database (created)
└── 📁 .venv/                   # Virtual environment (created)
```

---

## 🔌 LLM Providers

### OpenAI Provider

**Advantages:**
- ✅ Most accurate and capable
- ✅ Best at following complex instructions
- ✅ Built-in function calling support
- ✅ Managed vector search
- ✅ No local resources needed

**Disadvantages:**
- ❌ Costs money ($0.01-$0.15 per interaction)
- ❌ Requires internet connection
- ❌ Data sent to OpenAI servers
- ❌ Rate limits on free tier
- ❌ Subject to service outages

**Best For:**
- Production deployments
- When accuracy is critical
- When you have API credits
- Low-volume applications

### Ollama Provider

**Advantages:**
- ✅ Completely free
- ✅ Works offline
- ✅ Data stays on your machine
- ✅ No rate limits
- ✅ Customizable models

**Disadvantages:**
- ❌ Requires local hardware (4-16GB RAM)
- ❌ Slower responses on weak hardware
- ❌ Less accurate than GPT-4
- ❌ Manual function calling parsing
- ❌ Requires setup

**Best For:**
- Development and testing
- Privacy-sensitive applications
- High-volume usage
- Learning and experimentation
- Offline scenarios

---

## 📚 Knowledge Base

### Document Structure

The knowledge base consists of three Markdown files in the `data/` directory:

#### 1. **faq.md** (Frequently Asked Questions)
- General questions about orders
- Payment methods
- Account management
- Product information
- Contact information

#### 2. **shipping_policy.md** (Shipping Information)
- Processing times
- Shipping methods and costs
- Delivery timeframes
- International shipping
- Tracking information

#### 3. **returns_policy.md** (Returns & Refunds)
- Return eligibility
- Return process
- Refund timelines
- Exchange policy
- Exceptions and conditions

### Vector Store Options

#### OpenAI Vector Store
- **Storage**: Managed by OpenAI
- **Embedding Model**: text-embedding-ada-002
- **Search Quality**: Excellent
- **Cost**: $0.10/GB/month + search costs
- **Setup**: Run `seed_kb.py` → option 1

#### ChromaDB (Local)
- **Storage**: Local disk (./chroma_db)
- **Embedding Model**: all-MiniLM-L6-v2
- **Search Quality**: Good
- **Cost**: Free
- **Setup**: Run `seed_kb.py` → option 2

### Adding New Documents

```bash
# 1. Add markdown file to data/
echo "# New Policy\n\nContent here..." > data/new_policy.md

# 2. Reseed knowledge base
python seed_kb.py

# 3. Restart the agent
python app.py
```

### Document Formatting Tips

✅ **Do:**
- Use clear headings (##, ###)
- Write in FAQ format for common questions
- Include specific details (dates, prices, etc.)
- Use bullet points and lists
- Keep sections focused

❌ **Don't:**
- Mix unrelated topics in one file
- Use vague language
- Include very long paragraphs
- Forget to update after policy changes

---

## 🔧 Tools & Functions

### Available Tools

#### 1. `get_order_status`
**Purpose**: Look up order information by order ID

**Parameters:**
- `order_id` (string): Order ID like "ORD-1001"

**Returns:**
```json
{
  "found": true,
  "order_id": "ORD-1001",
  "status": "shipped",
  "tracking_number": "TRK123456789",
  "carrier": "BlueDart",
  "estimated_delivery": "2026-05-05"
}
```

**Mock Data:**
- ORD-1001: Shipped
- ORD-1002: Processing
- ORD-1003: Delayed

#### 2. `create_support_ticket`
**Purpose**: Create a support ticket for customer issues

**Parameters:**
- `issue_type` (string): Type of issue (e.g., "damaged_item")
- `order_id` (string): Related order ID
- `summary` (string): Description of the issue

**Returns:**
```json
{
  "ticket_id": "TICKET-0001",
  "issue_type": "damaged_item",
  "order_id": "ORD-1001",
  "summary": "Item arrived damaged",
  "status": "open"
}
```

### Adding New Tools

```python
# 1. Define function in tools.py
def my_new_tool(param1: str, param2: int) -> dict:
    """Tool description"""
    # Implementation
    return {"result": "success"}

# 2. Add to registry
TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "create_support_ticket": create_support_ticket,
    "my_new_tool": my_new_tool,  # Add here
}

# 3. Add tool definition to app.py (in provider classes)
# For OpenAI: Add to _get_tools()
# For Ollama: Add to _get_tool_prompt()
```

---

## 🔬 Advanced Usage

### Custom System Prompts

Edit `SYSTEM_PROMPT` in `app.py`:

```python
SYSTEM_PROMPT = """
You are a [YOUR CUSTOM ROLE].

You can:
- [CAPABILITY 1]
- [CAPABILITY 2]

Rules:
- [RULE 1]
- [RULE 2]
"""
```

### Switching Providers On-The-Fly

```bash
# In .env, change:
LLM_PROVIDER=openai   # or ollama

# Restart agent:
python app.py
```

### Using Different Ollama Models

```bash
# Pull model
ollama pull mistral

# Update .env
OLLAMA_MODEL=mistral:latest

# Restart
python app.py
```

### Programmatic Usage

```python
from app import OrderSupportAgent

agent = OrderSupportAgent()
response = agent.ask("Check order ORD-1001")
print(response)
```

### Conversation Export

```python
# In app.py, after conversation:
import json

# Save conversation
with open("conversation.json", "w") as f:
    json.dump(agent.conversation_history, f, indent=2)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Cannot import name 'OpenAI'"
```bash
# Solution: Install/reinstall openai
pip install --upgrade openai
```

#### 2. "Ollama connection refused"
```bash
# Solution: Start Ollama service
ollama serve

# Or check if different port:
OLLAMA_BASE_URL=http://localhost:11434
```

#### 3. "No vector store found"
```bash
# Solution: Run seeding script
python seed_kb.py
```

#### 4. "OpenAI quota exceeded"
```bash
# Solution: Switch to Ollama
# In .env:
LLM_PROVIDER=ollama
```

#### 5. "ChromaDB import error"
```bash
# Solution: Install chromadb
pip install chromadb sentence-transformers
```

#### 6. "Model not found" (Ollama)
```bash
# Solution: Pull the model
ollama pull llama3.2
```

### Debug Mode

```python
# Add to top of app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues

**Ollama is slow:**
- Use smaller model: `llama3.2:1b`
- Check CPU/RAM usage
- Ensure no other heavy processes

**OpenAI timeouts:**
- Check internet connection
- Try different model (gpt-3.5-turbo)
- Check OpenAI status page

---

## 💰 Performance & Costs

### OpenAI Costs (Estimates)

| Usage | Model | Tokens/Request | Cost/Request | Cost/Month* |
|-------|-------|----------------|--------------|-------------|
| Light | GPT-4.1 | ~2000 | $0.15 | $4.50 |
| Medium | GPT-4.1 | ~2000 | $0.15 | $45 |
| Heavy | GPT-4.1 | ~2000 | $0.15 | $450 |
| Light | GPT-3.5 | ~2000 | $0.002 | $0.06 |

*Assuming 1/10/100 requests per day

### Ollama Requirements

| Model | RAM | VRAM | Speed** | Quality |
|-------|-----|------|---------|---------|
| llama3.2:1b | 2GB | - | ~100 tok/s | Good |
| llama3.2:latest (8B) | 8GB | - | ~30 tok/s | Better |
| llama3.2:latest (8B, GPU) | 4GB | 6GB | ~100 tok/s | Better |

**On Apple M1/M2 chip

### Response Times

| Provider | Model | Avg Response | With Tools |
|----------|-------|--------------|------------|
| OpenAI | GPT-4 | 2-5s | 4-10s |
| OpenAI | GPT-3.5 | 1-2s | 2-4s |
| Ollama | Llama3.2 (CPU) | 5-15s | 10-30s |
| Ollama | Llama3.2 (GPU) | 2-5s | 4-10s |

---

## 💻 Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=app --cov=tools
```

### Code Style

```bash
# Install formatters
pip install black isort ruff

# Format code
black app.py tools.py seed_kb.py
isort app.py tools.py seed_kb.py

# Lint
ruff check .
```

### Type Checking

```bash
# Install mypy
pip install mypy

# Check types
mypy app.py tools.py
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add: my feature"

# Push and create PR
git push origin feature/my-feature
```

---

## 🚀 Production Deployment

### Checklist

- [ ] Replace mock data with real database
- [ ] Add authentication/authorization
- [ ] Implement proper logging
- [ ] Set up monitoring (Sentry, DataDog)
- [ ] Add rate limiting
- [ ] Configure CORS if web-facing
- [ ] Set up CI/CD pipeline
- [ ] Add backup strategy
- [ ] Document API endpoints
- [ ] Load test the system

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  agent:
    build: .
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./chroma_db:/app/chroma_db
    depends_on:
      - ollama
  
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### Web API (FastAPI)

```python
# api.py
from fastapi import FastAPI
from app import OrderSupportAgent

app = FastAPI()
agent = OrderSupportAgent()

@app.post("/chat")
def chat(message: str):
    response = agent.ask(message)
    return {"response": response}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution

- [ ] Add more LLM providers (Anthropic, Cohere, HuggingFace)
- [ ] Improve tool calling for Ollama
- [ ] Add web interface (Gradio/Streamlit)
- [ ] Add conversation memory/persistence
- [ ] Implement multi-language support
- [ ] Add more tools (refunds, exchanges)
- [ ] Improve error handling
- [ ] Add unit tests
- [ ] Optimize vector search

---

## 📄 License

MIT License - feel free to use this for any purpose.

---

## 🙏 Acknowledgments

- **OpenAI** for the GPT API
- **Ollama** for local LLM infrastructure
- **ChromaDB** for vector database
- **Rich** for beautiful terminal output
- **Sentence Transformers** for embeddings

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Questions**: Start a discussion
- **Email**: support@example.com (update this)

---

## 🗺️ Roadmap

### v2.1 (Current)
- ✅ OpenAI support
- ✅ Ollama support
- ✅ ChromaDB vector store
- ✅ Basic tools

### v2.2 (Next)
- [ ] Web interface
- [ ] Conversation persistence
- [ ] More LLM providers
- [ ] Improved tool parsing

### v3.0 (Future)
- [ ] Multi-language support
- [ ] Voice interface
- [ ] Integration with real APIs
- [ ] Analytics dashboard

---

**Made with ❤️ for the AI community**

*Last updated: May 2026*
