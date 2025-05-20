import os
from pipeline.preprocessing.slack_indexer import SlackIndexer
from pipeline.retrieval.database_connector import VectorSearcher

DB_PATH = "database"
SLACK_PATH = "data/slack/finance"

def main():
  # parses, indexes, and embeds slack messages into the database
  indexer = SlackIndexer(slack_dir = SLACK_PATH, db_path = DB_PATH)
  indexer.create_vector_store() # store the vector database

  #ask user what requests they want out of the RAG AI model
  while True:
    query = input("What can I help you with?\n")
    RAG_searcher = VectorSearcher(db_path = DB_PATH)
    top_k_results = RAG_searcher.search(query = query, top_k = 5) #searches and responds with top 5 k-map slack messages
    print(f"top 5 slack messages: {top_k_results}")



'''
class SlackSearchEngine:
    def __init__(self, slack_dir = "slack_messages", db_path = "chroma_db", model = "sentence-transformers/all-MiniLM-L6-v2"):
        self.slack_dir = slack_dir
        self.db_path = db_path
        self.model = model
        self.indexer = SlackIndexer(slack_dir, db_path, model)

    def setup(self):
        if not os.path.exists(self.db_path):
            print("Creating new vector store...")
            self.indexer.create_vectorstore()
        else:
            print("Loading existing vector store...")
            self.indexer.load_vectorstore()

    def run(self):
        self.setup()
        searcher = VectorSearcher(self.db_path, self.model)
        print("\n--- Slack Message Search Engine Ready ---")
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() == "exit":
                break
            results = searcher.search(query )
            if results:
                print("\nTop Matches:")
                for i, res in enumerate(results, 1):
                    print(f"\n[{i}] From: {res['filename']}")
                    print(f"    {res['text'][:200]}...")
            else:
                print("No relevant messages found.")





if __name__ == "__main__":
    engine = SlackSearchEngine()
    if not os.path.exists(engine.slack_dir):
        print(f"Error: Directory '{engine.slack_dir}' not found.")
    else:
        engine.run()
'''

if __name__ == "__main__":
  main()
