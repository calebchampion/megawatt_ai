'''
Searches a vector db in Chroma by using a huggingface anaylsis embedder
This module loads a persisted Chroma vector database and enables semantic
search over stored documents using a HuggingFace sentence transformer model.
'''

from typing import List, Dict
from langchain_community.vectorstores import Chroma #db used to store vectors
from langchain_community.embeddings import HuggingFaceEmbeddings #embedding query into vector to search
 
class VectorSearcher:
  def __init__(self, db_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):  #embedding model can change to many different ones to test out
    self.db_path = db_path
    self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")  #loads pretrained sentence embedding model (text->vectors) that capture meaning
    self.vectorstore = Chroma(persist_directory = db_path, embedding_function = self.embeddings)  #loads a chroma db from path

  def search(self, query: str, top_k: int = 5) -> List[Dict]:
    #takes a string, embeds it with embeddings model, then semanticly searches it with a k-map
    results = self.vectorstore.similarity_search(query, k = top_k)
    # returns a list of LangChain documents w/ .page_content and .metadata as i have described in slack_indexer
    #.similarity_search(query: str, k=5) — to find the k most similar documents
    #.as_retriever() — to use it in LangChain retrieval pipelines
    return [  #################################################################################at a later date update this ^^^^ as well may use retriever
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