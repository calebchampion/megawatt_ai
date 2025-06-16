'''
Parse the data, then index it into a vector database with Chromadb
Reads JSON files, chunks message data, embeds it 
using huggingface model, the stores the vectors in chroma
'''
import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings  #this model will eventually be updated to from fromlangchain_huggingface import HuggingFaceEmbeddings
#nothing else will change in the code
from langchain_community.vectorstores import Chroma #same with Chroma, it will throw a warning to a new model, it should be langchain-Chroma in the future
from langchain_core.documents import Document
from langchain_community.document_loaders import SlackDirectoryLoader
from collections import defaultdict

class SlackIndexer:
   #class for slack messages
   def __init__(self, slack_dir: str, db_path: str,
               embeddings, #model from huggingface, (text->vector) with semantic meaning
               chunk_size: int = 1000,  #customizable
               chunk_overlap: int = 200  #customizable
               ):

      self.slack_dir = slack_dir
      self.db_path = db_path
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.embeddings = embeddings
      self.vectorstore = None  #vector database placeholder


   #manual way to load in slack documents from unzipped folder
   def load_data(self) -> list[Document]:
      #loads Slack threads including replies from all channel folders inside slack_dir
      threads = {}  #thread_ts -> list of messages

      #iterate over each folder inside the main slack_dir
      for channel_folder in os.listdir(self.slack_dir):
         channel_path = os.path.join(self.slack_dir, channel_folder)
         if os.path.isdir(channel_path):
            for filename in os.listdir(channel_path):
               if filename.endswith(".json"):
                  file_path = os.path.join(channel_path, filename)
                  with open(file_path, "r", encoding = "utf-8") as f:  #open each json
                     messages = json.load(f)

                     for msg in messages:
                        thread_ts = msg.get("thread_ts") or msg.get("ts")
                        if thread_ts not in threads:
                           threads[thread_ts] = []
                        threads[thread_ts].append(msg)

      documents = []
      for thread_msgs in threads.values():
         filtered_msgs = [m for m in thread_msgs if "ts" in m]
         sorted_msgs = sorted(filtered_msgs, key = lambda m: m["ts"])
         thread_text = "\n".join(m.get("text", "") for m in sorted_msgs if m.get("text"))
         if thread_text.strip():
            documents.append(Document(page_content = thread_text))
      return documents
   '''
   
   def load_data(self) -> list[Document]:
      loader = SlackDirectoryLoader(self.slack_dir)
      flat_docs = loader.load()

      threads = defaultdict(list)
      for doc in flat_docs:
         thread_ts = doc.metadata.get("thread_ts") or doc.metadata.get("timestamp")
         if thread_ts:
            threads[thread_ts].append(doc)

      documents = []
      for thread_msgs in threads.values():
         sorted_docs = sorted(thread_msgs, key = lambda d: d.metadata.get("timestamp", ""))
         thread_text = "\n".join(d.page_content for d in sorted_docs if d.page_content.strip())
         if thread_text.strip():
            documents.append(Document(page_content = thread_text))

      print(f"Loaded grouped {len(documents)} documents from Slack export.")
      return documents
   '''
   def create_vector_store(self) -> Chroma:
      #creates a Chroma vector store from the text chunks with semantic embedding model
      raw_documents = self.load_data()

      print(f"\n\nLoaded {len(raw_documents)} documents\n\n")

      #splitting into chunks for vectors
      splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)
      documents = splitter.split_documents(raw_documents)

      #storing in database
      self.vectorstore = Chroma.from_documents(documents = documents, embedding = self.embeddings, persist_directory = self.db_path)  #load db to 
      self.vectorstore.persist()  #keep it on disk
      print(f"Vector store created and saved to {self.db_path}")

      return



class GoogleIndexer:
   #class for google emails
   def __init__(self, google_dir: str, db_path: str, embeddings, chunk_size: int = 1000, chunck_overlap: int = 200):
      self.google_dir = google_dir
      self.db_path = db_path
      self.embeddings = embeddings
      self.chunk_size = chunk_size
      self.chunk_overlap = chunck_overlap
      self.vector_store = None

   def add_vector_store(self):
      pass