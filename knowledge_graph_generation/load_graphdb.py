#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from configs import neo4j_host, neo4j_user, neo4j_password, neo4j_db, node_types, rel_types
from prompts import extract_triplets_template
from models import Llama3_2, Qwen2_5, Qwen2, DeepSeekR1Qwen15B
from neo4j import GraphDatabase
import time
import re

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory')
  flags.DEFINE_string('neo4j_db', default='testkg', help='neo4j database')
  flags.DEFINE_boolean('split', default = False, help = 'whether to split document')
  flags.DEFINE_enum('model', default = 'qwen2', enum_values = {'llama3', 'qwen2', 'deepseek'}, help = 'which LLM to use')

def main(unused_argv):
  driver = GraphDatabase.driver(neo4j_host, auth=(neo4j_user, neo4j_password))
  pattern = r'```json\s*(\{.*?\})\s*```|(\{.*?\})'
  llm = {
    'llama3': Llama3_2,
    # 'qwen2': Qwen2_5
    'qwen2': Qwen2,
    'deepseek': DeepSeekR1Qwen15B,
  }[FLAGS.model]()
  
  if FLAGS.split:
    text_splitter = RecursiveCharacterTextSplitter(separators = [r"\n\n", r"\n", r"\.(?![0-9])|(?<![0-9])\.", r"。"], is_separator_regex = True, chunk_size = 150, chunk_overlap = 10)
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext.lower() in ['.htm', '.html']:
        loader = UnstructuredHTMLLoader(join(root, f))
      elif ext.lower() == '.txt':
        loader = TextLoader(join(root, f))
      elif ext.lower() == '.pdf':
        loader = UnstructuredPDFLoader(join(root, f), mode = 'single')
      else:
        raise Exception('unknown format!')
      docs = loader.load()
      if FLAGS.split:
        docs = text_splitter.split_documents(docs)
      patent_text = "\n".join([doc.page_content for doc in docs])
      print(patent_text)
      
      prompt = """The given text is the text of a patent, extracted by OCR. Please extract the information of the patent in the following json format:

      {
        "patent_num": <patent id>,
        "patent_name": <patent name>,
        "applicant": <applicant name in string format>,
        "inventors": [<inventor1 in string format>, <inventor2 in string format>, ...],
        "assignee": <assignee name in string format>,
        "cited_patents": [<cited patent num1>, <cited patent num2>, ...],
        "fields of classification search": [<field1>, <field2>]
      }

      """
      try:
        response = llm.inference(patent_text, prompt)
        print(f"OCR+LLM: {response}")
      except Exception as e:
        print(f"OCR+LLM failed：{e}")

      info = None
      try:
        info = json.loads(response)
        print("Decoded patent info", json.dumps(info, indent=2, ensure_ascii=False))
      except Exception as e:
        print(f"JSONDecodeError: {e}")
        continue

      if not info or 'patent_num' not in info or 'patent_name' not in info or info['patent_num'] is None or info['patent_name'] is None:
        print("JSON incomplete，skip")
        continue

      print(f"✅ Saving into Neo4j: {info['patent_num']} - {info['patent_name']}")

      driver.execute_query('merge (a: Patent {patent_num: $pno, patent_name: $pnm}) return a;', pno=info['patent_num'],
                           pnm=info['patent_name'], database_=FLAGS.neo4j_db)
      if 'applicant' in info and info['applicant']:
        driver.execute_query('merge (a: Applicant {name: $name}) return a;', name=info['applicant'],
                             database_=FLAGS.neo4j_db)
        driver.execute_query(
          'match (a: Patent {patent_num: $pno}), (b: Applicant {name: $name}) merge (a)<-[:APPLY]-(b);',
          pno=info['patent_num'], name=info['applicant'], database_=FLAGS.neo4j_db)
      if 'inventors' in info and len(info['inventors']):
        for inventor in info['inventors']:
          driver.execute_query('merge (a: Inventor {name: $name}) return a;', name=inventor, database_=FLAGS.neo4j_db)
          driver.execute_query(
            'match (a: Patent {patent_num: $pno}), (b: Inventor {name: $name}) merge (a)<-[:INVENT]-(b);',
            pno=info['patent_num'], name=inventor, database_=FLAGS.neo4j_db)
      if 'assignee' in info and info['assignee']:
        driver.execute_query('merge (a: Assignee {name: $name}) return a;', name=info['assignee'],
                             database_=FLAGS.neo4j_db)
        driver.execute_query(
          'match (a: Assignee {name: $name}), (b: Patent {patent_num: $pno}) merge (a)<-[:ASSIGN]-(b);',
          name=info['assignee'], pno=info['patent_num'], database_=FLAGS.neo4j_db)
      if 'fields' in info and len(info['fields']):
        for field in info['fields']:
          driver.execute_query('merge (a: Field {name: $name}) return a;', name=field, database_=FLAGS.neo4j_db)
          driver.execute_query(
            'match (a: Patent {patent_num: $pno}), (b: Field {name: $name}) merge (a)-[:BELONGS_TO]-(b);',
            pno=info['patent_num'], name=field, database_=FLAGS.neo4j_db)

if __name__ == "__main__":
  start = time.time()
  add_options()

  try:
    app.run(main)
  finally:
    time_consumption = time.time() - start
    with open("time_consumption.txt", "w") as f:
      f.write(f"Time consumption: {time_consumption:.2f} s\n")
    print(f"Time consumption: {time_consumption:.2f} s (also saved to time_consumption.txt)")
