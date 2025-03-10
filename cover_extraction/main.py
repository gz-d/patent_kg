#!/usr/bin/python3

from absl import flags, app
from os import walk
from os.path import splitext, exists, join
import fitz # pymupdf
import json
import numpy as np
from models import Qwen25VL7B_dashscope, Qwen25VL7B_tgi, Qwen25VL7B_transformers
from configs import *

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to pdf')
  flags.DEFINE_enum('api', default = 'transformers', enum_values = {'dashscope', 'transformers', 'tgi'}, help = 'which api to call')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  for root, subFolders, files in walk(FLAGS.input_dir):
    for file in files:
      stem, ext = splitext(file)
      if ext != '.pdf': continue
      # 1) read first page of patent
      pdf = fitz.open(join(root, file))
      page = pdf[0]
      pix = page.get_pixmap(dpi=200)
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
      import pdb; pdb.set_trace()

if __name__ == "__main__":
  add_options()
  app.run(main)
