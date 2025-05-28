# Megawatt Onboarding AI Chatbot
An AI-powered onboarding assistant that helps new employees get up to speed by answering questions using company data from Slack messages, emails, and internal wiki pages.  Powered by a local RAG (Retrieval-Augmented Generation) pipeline with LangChain, Chroma, and an Ollama-hosted LLM.

## Architecture
- *[LangChain](https://www.langchain.com/)* – RAG orchestration
- *[ChromaDB](https://www.trychroma.com/)* – Local vector database
- *[Ollama](https://ollama.com/)* – Local LLM hosting (default -> LLaMA 3.2)
- *[HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)* - Embedding Model
- *[Flask](https://flask.palletsprojects.com/en/stable/)* – API
- *[Docker](https://www.docker.com/)* – Containerized deployment

### Main File
The main.py file runs everything together.
There are initial constants at the beggining like the database, slack, and google path.  These are used throughout, so they are easy to reconfigure.  The embedding model is also listed here and can be changed eaily as well.  (Take note if the embedding model changes, it may require you to also recalulate the entire database)
- **Indexer**
```python
class SlackIndexer:
   def __init__(self,
               slack_dir: str,
               db_path: str,
               embeddings,
               chunk_size: int = 1000,
               chunk_overlap: int = 200
               ):
```
- **Database Connector**
```python
class VectorSearcher:
  def __init__(self,
              db_path: str,
              embeddings
              ):
```
- **RAG Engine**
```python
class OllamaLLM:
    def __init__(self, model: str = "llama3.2"):
```
### Pipeline Folder
This folder has subfolders and a rag engine to do the heavy lifting and computation of the chatbot.
#### Preprocessing Folder
- **Embedding**: The indexer.py file inside this folder has two classes, one for Slack and one for Google.  These classes both go to the data/slack/ and data/google/ folders and read the given data into the program.  Then the data gets indexed and vectorized into a Chroma vector database.  The vectorization is done be a HuggingFace model that embeds the semantic meaning of text into the databse.
- **Storing**: The vector database is a Chromadb and is stored in database/ folder.  This database can then be seached through to find the nearest k-map of vectors from a query later on.
#### Retrievel Folder
This folder has a file called database_connector.py that can searches the database just above.  This also uses the embedding model to embed the query into a vector.  Then it uses a k-map to search for the closest corresponding results, fulfilling the retrieval part of the RAG mode.
#### RAG Engine File
This file rag_engine.py connects the two parts above to import the data, put it into a vector database, then query through it for searches.  The rag engine takes a query, give it to the retrieval to search for slack and google responses about the query, then promps the Ollama model to repond based on the retreived context and the initial query.  Then it prints out the reponse.

### Interface Folder
Flask and how it works

### Deploy Folder
Docker files and how it works
```bash
pip install -r requirements
```
## Setup Instructions
- Unzip slack data into data/slack folder and label it "mw"
- 
### Adding Data to the model
- Slack data: 
- Google data: 
