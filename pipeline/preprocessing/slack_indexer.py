'''
Parse the data, then index it into a vector database with Chromadb
Uses a huggingface model to embed the slack messages in a database
'''
import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # this model will eventually be updated to from fromlangchain_huggingface import HuggingFaceEmbeddings
#nothing else will change in the code
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class SlackIndexer:
   def __init__(
      self,
      slack_dir: str,
      db_path: str,
      embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", #model from huggingface, can replace with ollama it its better
      chunk_size: int = 1000,
      chunk_overlap: int = 200
   ):

      self.slack_dir = slack_dir
      self.db_path = db_path
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.embedding_model = embedding_model
      self.embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model) 
      self.vectorstore = None

   # load messages from slack into a list of dictionaries
   def load_messages(self) -> List[Dict]:
      all_messages = []
      for filename in os.listdir(self.slack_dir):
         if filename.endswith(".json"):
            path = os.path.join(self.slack_dir, filename)
            try:
               with open(path, "r", encoding = "utf-8") as f: # open the current JSON
                  data = json.load(f)

                  for message in data:
                     text = message.get("text", "").strip()
                     if not text:
                        continue
                     user = message.get("user_profile", {}).get("real_name") or message.get("user", "unknown")
                     ts = message.get("ts", "")
                     thread_ts = message.get("thread_ts", ts)
                     is_reply = "parent_user_id" in message

                     all_messages.append({
                        "text": text,
                        "filename": filename,
                        "user": user,
                        "ts": ts,
                        "thread_ts": thread_ts,
                        "is_reply": is_reply
                     })

            except (json.JSONDecodeError, FileNotFoundError) as e: #error reading into file
               print(f"Error loading {filename}: {e}")
      print(f"Read {len(all_messages)} files correctly into all_messages")  #checking
      return all_messages


   def chunk_texts(self, messages: List[Dict]) -> List[Dict]:
      # splits the loaded messages into smaller, more manageable chunks
      text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = self.chunk_size,
      chunk_overlap = self.chunk_overlap,
      length_function = len,
      )

      chunks = []
      for message in messages:
         text_chunks = text_splitter.split_text(message["text"])
         for chunk in text_chunks:
            chunks.append({
               "text": chunk,
               "filename": message["filename"],
               "user": message["user"],
               "ts": message["ts"],
               "thread_ts": message["thread_ts"],
               "is_reply": message["is_reply"]
            })
      return chunks
  
   # store in database of vectors
   def create_vector_store(self) -> Chroma:
      # creates a Chroma vector store from the text chunks
      if os.path.exists(os.path.join(self.db_path, "chroma.sqlite3")):
         print(f"Vector store already exists at {self.db_path}, skipping indexing.")
         self.vectorstore = Chroma(persist_directory = self.db_path, embedding_function = self.embeddings)
         return self.vectorstore

      print(f"Loading embedding model: {self.embedding_model}")
      #import as `from :class:`~langchain_huggingface import HuggingFaceEmbeddings
      #self.embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model)   updated embeddings 
      self.embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model)
      print("Creating vector store...")
      messages = self.load_messages()
      chunks = self.chunk_texts(messages)
      documents = [
         Document(
            page_content = chunk["text"],
            metadata = {
               "filename": chunk["filename"],
               "user": chunk["user"],
               "ts": chunk["ts"],
               "thread_ts": chunk["thread_ts"],
               "is_reply": chunk["is_reply"]
            }
         )
         for chunk in chunks
      ]

      self.vectorstore = Chroma.from_documents(documents = documents, embedding = self.embeddings, persist_directory = self.db_path,)
      self.vectorstore.persist()
      print(f"Vector store created and saved to {self.db_path}")

      return self.vectorstore
