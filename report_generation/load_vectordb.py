#!/usr/bin/python3

from absl import flags, app
from os import walk, environ
from os.path import exists, join, splitext
from tqdm import tqdm
import fitz # pymupdf
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from configs import *

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = 'patents', help = 'path to directory')
  flags.DEFINE_string('neo4j_db', default='patents', help='neo4j database')
  flags.DEFINE_string('chunk_dir', default = 'chunks', help = 'path to chunks')
  flags.DEFINE_string('doc_dir', default = 'docs', help = 'path to docs')

def main(unused_argv):
  environ['OCR_AGENT'] = 'tesseract'
  embedding = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  child_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
  chunk_vectordb = Neo4jVector(
    embedding = embedding,
    url = neo4j_host,
    username = neo4j_user,
    password = neo4j_password,
    database = FLAGS.neo4j_db,
    index_name = "chunk_vectordb",
    search_type = "hybrid",
    pre_delete_collection = True
  )
  chunk_store = LocalFileStore(FLAGS.chunk_dir)
  chunk_retriever = ParentDocumentRetriever(
    vectorstore = chunk_vectordb,
    docstore = create_kv_docstore(chunk_store),
    child_splitter = child_splitter,
    parent_splitter = parent_splitter
  )
  document_vectordb = Neo4jVector(
    embedding = embedding,
    url = neo4j_host,
    username = neo4j_user,
    password = neo4j_password,
    database = FLAGS.neo4j_db,
    index_name = "document_vectordb",
    search_type = "hybrid",
    pre_delete_collection = True
  )
  doc_store = LocalFileStore(FLAGS.doc_dir)
  doc_retriever = ParentDocumentRetriever(
    vectorstore = document_vectordb,
    docstore = create_kv_docstore(doc_store),
    child_splitter = child_splitter,
  )
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext != '.pdf': continue
      loader = UnstructuredPDFLoader(join(root, f), mode = 'single', strategy = 'hi_res', languages = ["chi_tra", "chi_sim", "eng"])
      docs = loader.load()
      if len(docs):
        chunk_retriever.add_documents(docs)
        doc_retriever.add_documents(docs)

if __name__ == "__main__":
  add_options()
  app.run(main)
