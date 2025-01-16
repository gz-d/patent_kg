#!/usr/bin/python3

from absl import flags, app
import fitz # pymupdf
import json
from PIL import Image
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to pdf')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  # 1) read first page of patent
  pdf = fitz.open(FLAGS.input)
  page = pdf[0]
  pix = page.get_pixmap(dpi=200)
  pix.save('cover.png')

  # 2) load tokenizer and LLM
  tokenizer = AutoTokenizer.from_pretrained(
    'THUDM/cogvlm2-llama3-chat-19B',
    trust_remote_code = True
  )
  torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
  model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm2-llama3-chat-19B',
    torch_dtype = torch_type
  ).to(FLAGS.device).eval()

  # 3) prepare text prompt
  messages = [
    {'role': 'user', 'content': """the given picture is the cover of a patent. please extract the applicant, inventors, assignee of the patent. please return only json code and in format:

{
  'applicant': <applicant name in string format>,
  'inventors': [<inventor1 in string format>, <inventor2 in string format>, ...],
  'assignee': <assignee name in string format>
}"""}
  ]
  text_prompt = tokenizer.apply_chat_template(
    messages, tokenize = False, add_generation_prompt = True
  )

  # 4) combine image prompt
  image = Image.open('cover.png').convert('RGB')
  encode = model.build_conversation_input_ids(
    tokenizer,
    query = text_prompt,
    history = list(),
    images = [image],
    template_version = 'chat'
  )
  inputs = {
    'input_ids': encode['input_ids'].unsqueeze(0).to(FLAGS.device),
    'token_type_ids': encode['token_type_ids'].unsqueeze(0).to(FLAGS.device),
    'attention_mask': encode['attention_mask'].unsqueeze(0).to(FLAGS.device),
    'images': [[encode['images'][0].to(FLAGS.device).to(torch_type)]]
  }

  # 5) generate
  with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens = 2048, pad_token_id = 128002)
    # remove input prompt, only leave the generated part
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0])
    response = response.split("<|end_of_text|>")[0]
  with open('results.json', 'w') as f:
    content = json.loads(response)
    f.write(json.dumps(content))

if __name__ == "__main__":
  add_options()
  app.run(main)
