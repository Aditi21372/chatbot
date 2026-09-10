# NeuroChat — RAG-Powered Chatbot API

> A production-style **Retrieval-Augmented Generation (RAG)** service that answers questions strictly from your own documents — built with FastAPI, LangChain, and FAISS.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0080FF)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-TinyLlama-FFD21E)

## What it does

1. **Ingests** any text document and splits it into overlapping chunks
2. **Embeds** the chunks with `sentence-transformers/all-MiniLM-L6-v2` into a FAISS vector index
3. **Retrieves** the most relevant chunks for a user query
4. **Generates** a grounded answer via an LLM — OpenAI GPT if a key is configured, or a fully **free local fallback** (TinyLlama-1.1B) so the service runs with zero API cost

Designed as an accessible, low-sensory assistant: answers are constrained to the provided context, keeping responses brief and factual.

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │                 FastAPI                  │
                        │        POST /chat      GET /health       │
                        └───────────────┬──────────────────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │      RetrievalQA Chain       │
                        │  (LangChain, grounded prompt) │
                        └───────┬──────────────┬───────┘
                                │              │
                    ┌───────────▼───┐   ┌──────▼──────────────┐
                    │  FAISS index  │   │         LLM          │
                    │  (MiniLM      │   │  OpenAI GPT, or      │
                    │   embeddings) │   │  TinyLlama fallback  │
                    └───────────┬───┘   └─────────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   DocumentProcessor      │
                    │   chunk 1000 / overlap 200│
                    └─────────────────────────┘
```

## Project structure

```
chatbot/
├── app.py                  # FastAPI app, chat + health endpoints
├── rag_chain.py            # RetrievalQA chain + custom LangChain LLM wrapper
├── embedding_indexer.py    # MiniLM embeddings → FAISS vector store
├── document_processor.py   # Document loading + text splitting
├── free_llm.py             # TinyLlama generation pipeline (free tier)
├── data/sample_text.txt    # Source document
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/Aditi21372/chatbot.git
cd chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Optional — to use GPT instead of the free model:

```bash
cp .env.example .env   # add your OPENAI_API_KEY
```

## API usage

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How can I improve productivity?"}'
```

```json
{
  "query": "How can I improve productivity?",
  "result": "..."
}
```

| Endpoint    | Method | Description                          |
|-------------|--------|--------------------------------------|
| `/chat`     | POST   | Ask a question, get a grounded answer |
| `/health`   | GET    | Service health check                 |

## Key engineering details

- **Custom LangChain `LLM` wrapper** (`FreeLLM`) integrates a Hugging Face `transformers` pipeline into LangChain's `RetrievalQA`, with Pydantic v2 compatibility
- **Graceful LLM fallback** — no API key needed to run the full pipeline
- **Grounded prompting** — the model is instructed to answer only from retrieved context and admit when it doesn't know

## Roadmap

- [ ] Swap in a proper vector DB (Qdrant/PGVector) for persistence
- [ ] Streaming responses over SSE
- [ ] Multi-document ingestion + source citations in responses

---

Built by [@Aditi21372](https://github.com/Aditi21372) · [More projects](https://github.com/Aditi21372?tab=repositories)
