# Megawatt Onboarding AI Chatbot
An AI-powered onboarding assistant that helps new employees get up to speed by answering questions using company data from Slack messages, emails, and internal wiki pages.  Powered by a local RAG (Retrieval-Augmented Generation) pipeline with LangChain, Chroma, and an Ollama-hosted LLM.

## Architecture
- *[LangChain](https://www.langchain.com/)* – RAG orchestration
- *[ChromaDB](https://www.trychroma.com/)* – Local vector database
- *[Ollama](https://ollama.com/)* – Local LLM hosting (default -> LLaMA 3.2)
- *[HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)* - Embedding Model
- *[Flask](https://flask.palletsprojects.com/en/stable/)* – API
- *[Docker](https://www.docker.com/)* – Containerized deployment

### Main
main file and how it works
### Pipeline
#### Preprocessing
Indexer files and how it works
#### Retrievel
Searcher and how it works
#### RAG Engine
RAG model file and how it works

### Interface
Flask and how it works

### Deploy
Docker files and how it works

## Setup Instructions
Docker stuff
