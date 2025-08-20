'''
Parse the data, then index it into a vector database with Chromadb
Reads JSON files, chunks message data, embeds it 
using huggingface model, the stores the vectors in chroma
'''
import os
import json
#from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings  #this model will eventually be updated to from fromlangchain_huggingface import HuggingFaceEmbeddings
#nothing else will change in the code
from langchain_community.vectorstores import Chroma #same with Chroma, it will throw a warning to a new model, it should be langchain-Chroma in the future
from langchain_core.documents import Document
#from langchain_community.document_loaders import SlackDirectoryLoader
#from collections import defaultdict

class SlackIndexer:
   #class for slack messages
   def __init__(self, slack_dir: str, db_path: str,
               embeddings, #model from huggingface, (text->vector) with semantic meaning
               chunk_size: int = 500,  #customizable
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
   
class TxtIndexer:
   #indexes txt info for updates
   def __init__(self, txt_dir: str, update_dir: str, db_path: str,
               embeddings: str, #model from huggingface, (text->vector) with semantic meaning
               chunk_size: int = 150,  #customizable
               chunk_overlap: int = 50  #customizable
               ):

      self.txt_dir = txt_dir
      self.update_dir = update_dir
      self.db_path = db_path
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.embeddings = embeddings
      self.vectorstore = None  #vector database placeholder

   def chunk_text(self, text, source_name):
      splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n"]
            )
      chunks = splitter.split_text(text)
      return [Document(page_content = chunk, metadata = {"source": source_name}) for chunk in chunks]
   
   def init_txt_file(self):
      # load DB
      db = Chroma(persist_directory = self.db_path, embedding_function = self.embeddings)

      # loop through .txt files
      for filename in os.listdir(self.txt_dir):
         if filename.endswith(".txt"):
            filepath = os.path.join(self.txt_dir, filename)
            with open(filepath, "r", encoding = "utf-8") as f:
               content = f.read()

            # langchain doc
            docs = self.chunk_text(content, filename)
                
            # add to db
            db.add_documents(docs)
            
      # persist changes
      db.persist()

   def add_text_files(self):
      # load DB
      db = Chroma(persist_directory = self.db_path, embedding_function = self.embeddings)

      # loop through .txt files
      for filename in os.listdir(self.update_dir):
         if filename.endswith(".txt"):
            filepath = os.path.join(self.update_dir, filename)
            with open(filepath, "r", encoding = "utf-8") as f:
               content = f.read()

            # langchain doc
            docs = self.chunk_text(content, filename)
                
            # add to db
            db.add_documents(docs)
            
            # del update files after
            os.remove(filepath)

            # persist changes
      db.persist()