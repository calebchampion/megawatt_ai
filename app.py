'''
main file to run everything together
'''
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

from pipeline.preprocessing.indexer import SlackIndexer, GoogleIndexer
from pipeline.retrieval.database_connector import VectorSearcher
from pipeline.rag_engine import OllamaLLM

#const
DB_PATH = "database"
SLACK_PATH = "data/slack/mw_slack"
UPDATE_SLACK_PATH = "data/update/mw_slack"
GOOGLE_PATH = "data/google/mw_google"
UPDATE_GOOGLE_PATH = "data/update/mw_google"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  #sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL = "gemma3:1b"  ## gemma3:1b   qwen2.5:3b   llama3.2

# --- Cached resources ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_searcher():
    return VectorSearcher(db_path = DB_PATH, embeddings = load_embeddings())

@st.cache_resource
def load_llm():
    return OllamaLLM(model = LLM_MODEL)

#optional indexing if no DB exists
def index_if_needed():
    if not os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")):
        st.info("No database found. Indexing Slack data now...")
        indexer = SlackIndexer(slack_dir = SLACK_PATH, db_path = DB_PATH, embeddings = load_embeddings())
        indexer.create_vector_store()
        st.success("✅ Indexing complete!")
    else:
        st.success("Starting...")



#main app UI
def main():
    st.set_page_config(page_title = "MW AI Assistant", layout = "wide")
    st.title("🧠 Megawatt Assistant")


    # preload caches
    with st.spinner("Initializing AI Assistant... Don't refresh"):
        load_embeddings()
        index_if_needed()
        searcher = load_searcher()
        llm = load_llm()

    query = st.text_input("Ask a question", placeholder = "e.g., What does Wayne do?")
    if st.button("Ask") and query:

        with st.spinner("Searching knowledge base..."):
            results = searcher.search(query = query, top_k = 15)

        with st.spinner("Generating response..."):
            answer = llm.generate_with_context(query = query, recieved_data = results)

        st.subheader("💬 AI Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()