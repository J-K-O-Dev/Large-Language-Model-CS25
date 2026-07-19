# Week 8 - RAG Chatbot With Flask And React

Week 8 turned the project from notebooks into an interactive application. The goal was to build a Retrieval-Augmented Generation chatbot that can answer with context from local LLM research papers.

The backend handles loading, chunking, embedding, retrieval, and generation. The frontend gives the user a simple chat interface with persistent history.

## Goal

Build a full-stack local RAG chatbot over a folder of documents.

## Files

| File or folder | Purpose |
| --- | --- |
| `notebook/RAG_Pipe_Line.ipynb` | Original RAG pipeline notebook |
| `notebook/app.py` | Flask backend with `/chat` endpoint |
| `data/` | Local PDF knowledge base and persistent ChromaDB store |
| `react_interface/` | React frontend |
| `react_interface/package.json` | Frontend dependencies and npm scripts |

## What Was Implemented

- Multi-file document loader for `.pdf`, `.docx`, `.txt`, `.csv`, and `.json`.
- Recursive text chunking with LangChain splitters.
- Sentence embeddings with `all-MiniLM-L6-v2`.
- Persistent ChromaDB vector storage in `data/vector_store/`.
- Retrieval of the top relevant chunks for a user query.
- TinyLlama generation through Hugging Face `transformers`.
- Flask API endpoint:
  - `POST /chat`
  - accepts conversation history
  - retrieves context for the latest user message
  - builds an augmented prompt
  - returns a generated reply
- React chat interface with:
  - user and assistant message bubbles
  - loading state
  - disabled input while generation runs
  - localStorage-backed chat history
  - clear chat action

## Result

The RAG notebook loaded `227` document pages from the six local PDFs in `data/`. The backend builds or reuses a ChromaDB vector store, retrieves relevant chunks, and passes that context into TinyLlama before generation.

The result is a usable local chatbot that can ground its answers in the provided LLM papers instead of relying only on model weights.

## How To Run

Install Python dependencies from the root first:

```bash
pip install -r requirements.txt
```

Start the backend from the `notebook` directory so relative paths resolve correctly:

```bash
cd Week8_work\notebook
python app.py
```

The backend runs on:

```text
http://localhost:5000
```

Start the React frontend in a second terminal:

```bash
cd Week8_work\react_interface
npm install
npm start
```

The frontend usually opens on:

```text
http://localhost:3000
```

## Requirements

Python packages:

- `flask`
- `flask-cors`
- `torch`
- `transformers`
- `sentence-transformers`
- `chromadb`
- `langchain-community`
- `langchain-core`
- `langchain-text-splitters`
- `pypdf`
- `docx2txt`
- `tiktoken`
- `scikit-learn`

Frontend requirements:

- Node.js
- npm
- React dependencies from `react_interface/package.json`

## Learning From The PoA

The PoA explains RAG as a way to move beyond closed-book generation. Instead of expecting the LLM to store every fact inside its weights, the system retrieves relevant external documents at inference time and places them into the prompt.

The core learning was the full retrieval loop: load documents, split them into chunks, embed each chunk, store vectors, retrieve the top matching chunks, and generate with grounded context. This is also the base layer for more agentic systems, where an LLM can call tools, run code, or search through external resources in multiple steps.

## Takeaway

RAG made the language model feel much more useful. Instead of asking the model to remember everything, the system retrieves relevant source text at query time and gives the model better context for answering.
