#!/usr/bin/python3

from absl import flags, app
from os import walk
from os.path import splitext, exists, join
import fitz # pymupdf
import json
import re
import numpy as np
from models import Qwen25VL7B_dashscope, Qwen25VL7B_tgi, Qwen25VL7B_transformers, DeepSeek_VL2_transformers
from neo4j import GraphDatabase
from configs import *
import time

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to pdf')
  flags.DEFINE_string('neo4j_db', default='testkg', help='neo4j database')
  flags.DEFINE_enum('api', default = 'transformers', enum_values = {'dashscope', 'transformers', 'tgi', 'deepseek'}, help = 'which api to call')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')


def extract_json(text):
  matches = re.findall(r"\{.*?\}", text, re.DOTALL)
  if matches:
    json_str = matches[-1] 
    try:
      return json.loads(json_str)
    except json.JSONDecodeError as e:
      print(f"⚠️ JSONDecodeError {e}\nTry repairing JSON")
      return None
  return None

def main(unused_argv):
  driver = GraphDatabase.driver(neo4j_host, auth = (neo4j_user, neo4j_password))
  # pattern = r'```json(.*?)```|```(.*?)```'
  pattern = r'```json\s*(\{.*?\})\s*```|(\{.*?\})'
  for root, subFolders, files in walk(FLAGS.input_dir):
    for file in files:
      stem, ext = splitext(file)
      if ext != '.pdf': continue
      # 1) read first page of patent
      pdf = fitz.open(join(root, file))
      page = pdf[0]
      pix = page.get_pixmap(dpi=100)
      img = np.frombuffer(pix.samples, dtype = np.uint8).reshape(pix.height, pix.width, -1)
      if pix.n == 1: img = img[:,:,0]
      elif pix.n == 3: img = img.reshape(pix.height, pix.width, 3)
      elif pix.n == 4: img = img.reshape(pix.height, pix.width, 4)
      # 2) call vqa model to extract
      if FLAGS.api == 'transformers':
        model = Qwen25VL7B_transformers(huggingface_token)
      elif FLAGS.api == 'dashscope':
        model = Qwen25VL7B_dashscope(dashscope_api_key)
      elif FLAGS.api == 'tgi':
        model = Qwen25VL7B_tgi(tgi_host)
      elif FLAGS.api == 'deepseek':
        model = DeepSeek_VL2_transformers(huggingface_token)
      else:
        raise Exception('unknown api!')
      prompt = """the given picture is the cover of a patent. please extract the information of the patent in the following json format:

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
        response = model.inference(prompt, img)
        print(f"VQA: {response}")
      except Exception as e:
        print(f"VQA failed：{e}")

      if FLAGS.api != 'deepseek':
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches) < 1:
          print("Couldn't find valid JSON")
          continue

        info = None
        try:
          info = json.loads(matches[0][0])
          print("Decoded patent information", json.dumps(info, indent=2, ensure_ascii=False))
        except Exception as e:
          print(f"JSONDecodeError: {e}")
          continue

      else:
        info = extract_json(response)

      print(f"INFO: {info}")

      if not info or 'patent_num' not in info or 'patent_name' not in info or info['patent_num'] is None or info['patent_name'] is None:
        print("JSON not complete，skip")
        continue

      print(f"✅ Saving into Neo4j: {info['patent_num']} - {info['patent_name']}")

      driver.execute_query('merge (a: Patent {patent_num: $pno, patent_name: $pnm}) return a;', pno = info['patent_num'], pnm = info['patent_name'], database_ = FLAGS.neo4j_db)
      if 'applicant' in info and info['applicant']:
        driver.execute_query('merge (a: Applicant {name: $name}) return a;', name = info['applicant'], database_ = FLAGS.neo4j_db)
        driver.execute_query('match (a: Patent {patent_num: $pno}), (b: Applicant {name: $name}) merge (a)<-[:APPLY]-(b);', pno = info['patent_num'], name = info['applicant'], database_ = FLAGS.neo4j_db)
      if 'inventors' in info and len(info['inventors']):
        for inventor in info['inventors']:
          driver.execute_query('merge (a: Inventor {name: $name}) return a;', name = inventor, database_ = FLAGS.neo4j_db)
          driver.execute_query('match (a: Patent {patent_num: $pno}), (b: Inventor {name: $name}) merge (a)<-[:INVENT]-(b);', pno = info['patent_num'], name = inventor, database_ = FLAGS.neo4j_db)
      if 'assignee' in info and info['assignee']:
        driver.execute_query('merge (a: Assignee {name: $name}) return a;', name = info['assignee'], database_ = FLAGS.neo4j_db)
        driver.execute_query(
          'match (a: Assignee {name: $name}), (b: Patent {patent_num: $pno}) merge (a)<-[:ASSIGN]-(b);',
          name=info['assignee'], pno=info['patent_num'], database_=FLAGS.neo4j_db)
      if 'fields' in info and len(info['fields']):
        for field in info['fields']:
          driver.execute_query('merge (a: Field {name: $name}) return a;', name = field, database_ = FLAGS.neo4j_db)
          driver.execute_query('match (a: Patent {patent_num: $pno}), (b: Field {name: $name}) merge (a)-[:BELONGS_TO]-(b);', pno = info['patent_num'], name = field, database_ = FLAGS.neo4j_db)

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
