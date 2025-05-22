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

class SlackIndexer:
   def __init__(self, slack_dir: str, db_path: str,
               embeddings, #model from huggingface, (text->vector) with semantic meaning
               google_dir: str = "data/google",
               chunk_size: int = 1000,  #customizable
               chunk_overlap: int = 200  #customizable
               ):

      self.slack_dir = slack_dir
      self.google_dir = google_dir
      self.db_path = db_path
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.embeddings = embeddings
      self.vectorstore = None  #vector database placeholder

  
   def create_vector_store(self) -> Chroma:
      #creates a Chroma vector store from the text chunks with semantic embedding model
      loader = SlackDirectoryLoader(self.slack_dir)
      raw_documents = loader.load()
      print(f"\n\nLoaded {len(raw_documents)} documents\n\n")

      splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)
      documents = splitter.split_documents(raw_documents)

      self.vectorstore = Chroma.from_documents(documents = documents, embedding = self.embeddings, persist_directory = self.db_path)  #load db to 
      self.vectorstore.persist()  #keep it on disk, in this version, i think it persists it anyways
      print(f"Vector store created and saved to {self.db_path}")

      return
   


class GoogleIndexer:
   def __init__(self):
      pass 