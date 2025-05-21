'''
Parse the data, then index it into a vector database with Chromadb
Reads JSON files, chunks message data, embeds it 
using huggingface model, the stores the vectors in chroma
'''

import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  #this model will eventually be updated to from fromlangchain_huggingface import HuggingFaceEmbeddings
#nothing else will change in the code
from langchain_community.vectorstores import Chroma #same with Chroma, it will throw a warning to a new model, it should be langchain-Chroma in the future
from langchain_core.documents import Document

class SlackIndexer:
   def __init__(self, slack_dir: str, db_path: str,
               embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", #model from huggingface, (text->vector) with semantic meaning
               chunk_size: int = 1000,  #customizable
               chunk_overlap: int = 200  #customizable
               ):

      self.slack_dir = slack_dir
      self.db_path = db_path
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.embedding_model = embedding_model
      self.embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model)  #loads a pretrained model for semantic text emmbedding -> vectors
      self.vectorstore = None  #vector database placeholder

   def load_messages(self) -> List[Dict]:
      #load messages from slack into a list of dictionaries
      all_messages = []

      for subdir in os.listdir(self.slack_dir):
         if subdir.endswith(".json"):
            continue
         subdir_path = os.path.join(self.slack_dir, subdir)
         if os.path.isdir(subdir_path):  # Ensure it's a folder
            for filename in os.listdir(subdir_path):
               if filename.endswith(".json"):
                  path = os.path.join(subdir_path, filename)
                  try:
                     with open(path, "r", encoding="utf-8") as f:
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
                              "filename": os.path.relpath(path, self.slack_dir),
                              "user": user,
                              "ts": ts,
                              "thread_ts": thread_ts,
                              "is_reply": is_reply
                           })
                  except (json.JSONDecodeError, FileNotFoundError) as e:
                     print(f"Error loading {filename}: {e}")
         print(f"Folder {subdir_path} done")
      print(f"\nRead {len(all_messages)} messages from subfolders.")
      return all_messages


   def chunk_texts(self, messages: List[Dict]) -> List[Dict]:
      #splits the loaded messages into smaller, more manageable chunks
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
  
   def create_vector_store(self) -> Chroma:
      #creates a Chroma vector store from the text chunks with semantic embedding model
      print(f"Loading embedding model: {self.embedding_model}")
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

      self.vectorstore = Chroma.from_documents(documents = documents, embedding = self.embeddings, persist_directory = self.db_path)  #load db to 
      self.vectorstore.persist()  #keep it on disk, in this version, i think it persists it anyways
      print(f"Vector store created and saved to {self.db_path}")

      return