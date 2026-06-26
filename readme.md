# 📘 RAG-with-PDF — Chat with your PDF

A lightweight Retrieval-Augmented Generation (RAG) app that lets you upload any PDF and ask questions about it in a chat interface. Every answer is grounded in the document's own text, and each response is tagged with the exact page numbers it was pulled from.

## Features

- Drag-and-drop PDF upload, indexed in seconds
- Chunking + embedding with FAISS for fast similarity search
- Fast LLM inference via Groq (Llama 3.3 70B)
- Page-level citation chips under every answer (e.g. `p.3 · p.7`)
- Per-document chat history, with options to clear the conversation or swap documents
- Custom-themed UI — no default Streamlit styling

## Tech stack

| Layer        | Tool                                              |
|--------------|----------------------------------------------------|
| UI           | [Streamlit](https://streamlit.io)                  |
| Orchestration| [LangChain](https://www.langchain.com)             |
| LLM          | [Groq](https://groq.com) — `llama-3.3-70b-versatile` |
| Embeddings   | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| PDF parsing  | `PyPDFLoader` (LangChain Community)                 |

## How it works

1. Upload a PDF from the sidebar.
2. The app splits it into ~1000-character chunks (200-character overlap) and embeds each chunk.
3. On each question, it retrieves the 4 most relevant chunks via similarity search.
4. The LLM is instructed to answer **only** from that retrieved context — if the context doesn't cover it, it says so instead of guessing.
5. The page numbers behind the retrieved chunks are shown as citation chips under the answer.

## Setup

```bash
git clone https://github.com/Bunny-777/RAG-with-PDF
cd RAG-with-PDF
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

## Run it

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
├── .env               # Your API key (not committed — see .gitignore)
└── README.md
```

## Notes

- First run will download the embedding model (~80MB) and may take a moment.
- The vector index is rebuilt whenever you upload a new PDF; it's cached for the session otherwise.

