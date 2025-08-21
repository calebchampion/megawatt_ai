# Megawatt Onboarding AI Chatbot
> An AI-powered onboarding assistant that helps new employees get up to speed by answering questions using company data from Slack messages, emails, and internal wiki pages.  Powered by a local RAG (Retrieval-Augmented Generation) pipeline with LangChain, Chroma, and an Ollama-hosted LLM.

## Architecture
![Architecture](architecture.png)

**Overview** (*How it works*): 
1. Company data is preprocessed and embedded into a local vector database.
2. A user submits a query to the chatbot.
3. The chatbot embeds the query and performs a similarity search in the vector database.
4. The retrieved context and query are passed to a local LLM (Ollama) to generate an accurate and context-aware response.


**Tools Used**
- **Embedding & Vector Store**: [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [ChromaDB](https://www.trychroma.com/)
- **RAG Orchestration**: [LangChain](https://www.langchain.com/)
- **Local LLM**: [Ollama](https://ollama.com/)
- **UI**: [Streamlit](https://streamlit.io/)

### Main File
The `app.py` file runs everything together.
There are initial constants at the beginning of the file like the database, slack, and google path.  These are used throughout, so they are easy to reconfigure.  The embedding model is also listed here and can be changed easily as well.  (Take note if the embedding model changes, it may require you to also recalculate the entire database).  I've listed the classes I've made and used here in the main file.
- **Indexer** : Indexes and places the slack & text data into the database.
```python
class SlackIndexer:
   def __init__(self,
               slack_dir: str,
               db_path: str,
               embeddings,
               chunk_size: int = 800,
               chunk_overlap: int = 200
               ):
```
```python
class TxtIndexer:
   def __init__(self,
              txt_dir: str,
              update_dir: str,
              db_path: str,
              embeddings: str,
              chunk_size: int = 200, 
              chunk_overlap: int = 50  
              ):
```
- **Database Connector** : Searches through the database to find best material for the LLM.
```python
class VectorSearcher:
  def __init__(self,
              db_path: str,
              embeddings
              ):
```
- **RAG Engine** : Connects everything together with the LLM.
```python
class OllamaLLM:
    def __init__(self, model: str = "gemma3:1b"):
```
### Pipeline Folder
This folder has subfolders and a rag engine to do the heavy lifting and computation of the chatbot.
###### Preprocessing Folder
- **Embedding**: The `indexer.py` file inside this folder has two classes, one for Slack and one for Google.  These classes both go to the `data/slack/` and `data/google/` folders and read the given data into the program.  Then the data gets indexed and vectorized into a Chroma vector database.  The vectorization is done by a HuggingFace model that embeds the semantic meaning of text into the database.
- **Storing**: The vector database is a Chromadb and is stored in `database/folder`.  This database can then be searched through to find the nearest k-map of vectors from a query later on.
###### Retrievel Folder
- `database_connector.py` handles vector similarity search.
- The query is embedded and used to retrieve the top-k most relevant vectors using cosine similarity.
###### RAG Engine File
- `rag_engine.py` connects everything: it receives a query, retrieves relevant context, and prompts the Ollama LLM to return a final answer.

## Installation on Computer (Already Done)
- Assuming pip, python, venv, & program folders are on computers
- Virtual environment (venv) is already set up
- Download ollama, and run ```ollama run gemma3:1b```
- Unzip slack data into `data/slack` folder and name `mw_slack`
- Activate the venv already on project by running the command```source venv/bin/activate``` under the `AI_Project folder`
- In the venv run the command ```pip install -r requirements.txt```
- In the termimal of the project with the venv activated, place the command ```streamlit run app.py```, this will take a while to start up the first time you do it ~10 min, and less time to initially start up after that.

## Setup
- Log into the ubuntu computer, username > megawatt, password > megawatt
- Go to folder `AI_Project` and run command in the terminal ```source venv/bin/activate``` and then ```streamlit run app.py```
- This will update everthing and launch a local web address to the AI model.
- Now you can go to that web IP address to access the model from any machine.

#### Updating data to the model
*Slack data*
- Just unzip the slack zip file and place it in `data/slack` folder as labeled `mw_slack`.
- Then delete all the files from the `database` folder.
- Rerun the program with ```streamlit run app.py```, this will take a while.


*Text data*:
- Restart the computer so the program is not running, make a text document in `data/update` then add whatever you want to it.
- Run the program by ```streamlit run app.py```, and it will automatically update, run the model, & delete the text file when it's done.