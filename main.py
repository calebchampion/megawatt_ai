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
SLACK_PATH = "data/slack/Megawatt Slack export May 1 2019 - May 19 2025"

#main
def main():
  '''load preloaded ai embedding model
  '''
  embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")



  '''parses and indexes into vector db if needed
  '''
  if os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")): 
    print("\nDatabase already exists. Skipping indexing.\n")

  else: #make a database for it
    indexer = SlackIndexer(slack_dir = SLACK_PATH, db_path = DB_PATH, embeddings = embedding_model)
    indexer.create_vector_store() # store the vector database



  '''query and prompt ai model
  '''
  RAG_searcher = VectorSearcher(db_path = DB_PATH, embeddings = embedding_model)  #load model object in beforehand so it doesnt load every time
  while True:
    query = input("What can I help you with? -> ")
    if query.lower() == "exit":
      break
    top_k_results = RAG_searcher.search(query = query, top_k = 20) #searches and responds with top 10 k-map slack messages
    print(f"\ntop 5 slack messages: \n\n{top_k_results}")

    LLM = OllamaLLM()  #pick from any model model = "llama3.2" as default
    stream = LLM.generate_with_context(query = query, recieved_data = top_k_results)



#run
if __name__ == "__main__":
  main()

