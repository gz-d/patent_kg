#!/usr/bin/python3

from absl import flags, app
from os import walk, environ
from os.path import exists, join, splitext
from tqdm import tqdm
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from configs import *

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = 'patents', help = 'path to directory')

def main(unused_argv):
  environ['OCR_AGENT'] = 'tesseract'
  embedding = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  vectordb = Neo4jVector(
    embedding = embedding,
    url = neo4j_host,
    username = neo4j_user,
    password = neo4j_password,
    database = neo4j_db,
    index_name = "typical_rag",
    search_type = "hybrid",
    pre_delete_collection = True
  )
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext != '.pdf': continue
      loader = UnstructuredPDFLoader(join(root, f), mode = 'single', strategy = 'hi_res', languages = {'en', 'zh-cn', 'zh-tw'})
      docs = loader.load()
      for page_num, doc in enumerate(docs):
        doc.metadata['page_num'] = page_num
        doc.metadata['patent_path'] = join(root, f)
      vectordb.add_documents(docs)

if __name__ == "__main__":
  add_options()
  app.run(main)

