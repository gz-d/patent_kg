#!/usr/bin/python3

from absl import flags, app
from os import walk
from os.path import splitext, exists, join
import fitz # pymupdf
import json
import re
import numpy as np
from models import Qwen25VL7B_dashscope, Qwen25VL7B_tgi, Qwen25VL7B_transformers
from neo4j import GraphDatabase
from configs import *

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to pdf')
  flags.DEFINE_enum('api', default = 'transformers', enum_values = {'dashscope', 'transformers', 'tgi'}, help = 'which api to call')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  driver = GraphDatabase.driver(neo4j_host, auth = (neo4j_user, neo4j_password))
  pattern = r'```json(.*?)```|```(.*?)```'
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
        model = Qwen25VL7B_transformers(huggingface_api_key)
      elif FLAGS.api == 'dashscope':
        model = Qwen25VL7B_dashscope(dashscope_api_key)
      elif FLAGS.api == 'tgi':
        model = Qwen25VL7B_tgi(tgi_host)
      else:
        raise Exception('unknown api!')
      response = model.inference("""the given picture is the cover of a patent. please extract the information of the patent in the following json format:

{
  "patent_num": <patent id>,
  "patent_name": <patent name>,
  "applicant": <applicant name in string format>,
  "inventors": [<inventor1 in string format>, <inventor2 in string format>, ...],
  "assignee": <assignee name in string format>,
  "cited_patents": [<cited patent num1>, <cited patent num2>, ...],
  "fields of classification search": [<field1>, <field2>]
}""", img)
      matches = re.findall(pattern, response, re.DOTALL)
      if len(matches) < 1: continue
      try:
        info = json.loads(matches[0][0])
      except:
        continue
    driver.execute_query('merge (a: Patent {patent_num: $pno, patent_name: $pnm}) return a;', pno = info['patent_num'], pnm = info['patent_name'], database_ = neo4j_db)
    if 'applicant' in info:
      driver.execute_query('merge (a: Applicant {name: $name}) return a;', name = info['applicant'], database_ = neo4j_db)
      driver.execute_query('match (a: Patent {patent_num: $pno}), (b: Applicant {name: $name}) merge (a)<-[:APPLY]-(b);', pno = info['patent_num'], pnm = info['patent_name'], database_ = neo4j_db)
    if 'inventors' in info and len(info['inventors']):
      for inventor in info['inventors']:
        driver.execute_query('merge (a: Inventor {name: $name}) return a;', name = inventor, database_ = neo4j_db)
        driver.execute_query('match (a: Patent {patent_num: $pno}), (b: Inventor {name: $name}) merge (a)<-[:INVENT]-(b);', pno = info['patent_num'], name = inventor, database_ = neo4j_db)
    if 'assignee' in info:
      driver.execute_query('merge (a: Assignee {name: $name}) return a;', name = info['assignee'], database_ = neo4j_db)
    if 'fields' in info and len(info['fields']):
      for field in info['fields']:
        driver.execute_query('merge (a: Field {name: $name}) return a;', name = field, database_ = neo4j_db)
        driver.execute_query('match (a: Patent {patent_num: $pno}), (b: Field {name: $name}) merge (a)-[:BELONGS_TO]-(b);', pno = info['patent_num'], name = field, database_ = neo4j_db)

if __name__ == "__main__":
  add_options()
  app.run(main)
