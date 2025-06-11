'''
main file to run everything together
'''

import os
from langchain_community.embeddings import HuggingFaceEmbeddings

from pipeline.preprocessing.indexer import SlackIndexer, GoogleIndexer
from pipeline.retrieval.database_connector import VectorSearcher
from pipeline.rag_engine import OllamaLLM

#const
DB_PATH = "database"
SLACK_PATH = "data/slack/mw_slack"
UPDATE_SLACK_PATH = "data/update/mw_slack"
GOOGLE_PATH = "data/google/mw_google"
UPDATE_GOOGLE_PATH = "data/update/mw_google"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")  #sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL = "gemma3:1b"  ## gemma3:1b   qwen2.5:3b   llama3.2

#main
def main():
  '''
  parses and indexes into vector db if needed
  '''
  if os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")):
    print("\nDatabase already exists. Skipping indexing.\n")

  else: #make a database for it
    indexer = SlackIndexer(slack_dir = SLACK_PATH, db_path = DB_PATH, embeddings = EMBEDDING_MODEL)  #optional chunk size and overlap parameters as well
    indexer.create_vector_store() # store the vector database


  '''
  query and prompt ai model
  '''
  RAG_searcher = VectorSearcher(db_path = DB_PATH, embeddings = EMBEDDING_MODEL)   #classes objects so they dont load every time a questions asked
  LLM = OllamaLLM(model = LLM_MODEL)  #pick from any model model = "llama3.2" as default, qwen2.5:3b

  while True:
    query = input("\n\nWhat can I help you with? -> ")
    if query.lower() == "exit":
      break

    top_k_results = RAG_searcher.search(query = query, top_k = 15)  #searches and responds with top 10 k-map slack messages

    #names
    LLM.generate_with_context(query = query, recieved_data = top_k_results)


#run
if __name__ == "__main__":
  main()