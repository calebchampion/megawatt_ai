from typing import List, Dict
from langchain_community.vectorstores import Chroma #db used to store vectors
from langchain_community.embeddings import HuggingFaceEmbeddings #embedding query into vector to search


class VectorSearcher:
  def __init__(self, db_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    self.db_path = db_path
    self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    self.vectorstore = Chroma(persist_directory = db_path, embedding_function = self.embeddings)

  def search(self, query: str, top_k: int = 5) -> List[Dict]:
    results = self.vectorstore.similarity_search(query, k = top_k)
    return [
            {
              "text": doc.page_content,
              "filename": doc.metadata.get("filename", "unknown"),
              "user": doc.metadata.get("user", "unknown"),
              "ts": doc.metadata.get("ts", ""),
              "thread_ts": doc.metadata.get("thread_ts", ""),
              "is_reply": doc.metadata.get("is_reply", False)
            }
          for doc in results
    ]