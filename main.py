'''
main file to run everything together
'''

import os
from pipeline.preprocessing.slack_indexer import SlackIndexer
from pipeline.retrieval.database_connector import VectorSearcher
from pipeline.rag_engine import OllamaLLM

#const
DB_PATH = "database"
SLACK_PATH = "data/slack/Megawatt Slack export May 1 2019 - May 19 2025"

#main
def main():
  #parses, indexes, and embeds slack messages into the database
  #if there is a database already
  if os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")): 
    print("\nDatabase already exists. Skipping indexing.\n")

  else: #make a database for it
    indexer = SlackIndexer(slack_dir = SLACK_PATH, db_path = DB_PATH)
    indexer.create_vector_store() # store the vector database

  #ask user what requests they want out of the RAG AI model
  RAG_searcher = VectorSearcher(db_path = DB_PATH)  #load model object in beforehand so it doesnt load every time
  while True:
    query = input("What can I help you with? -> ")
    if query.lower() == "exit":
      break
    top_k_results = RAG_searcher.search(query = query, top_k = 20) #searches and responds with top 10 k-map slack messages
    print(f"\ntop 5 slack messages: \n\n{top_k_results}")

    LLM = OllamaLLM()  #pick from any model model = "llama3.2" as default
    stream = LLM.generate_with_context(query = query, top_5 = top_k_results)

#run
if __name__ == "__main__":
  main()
