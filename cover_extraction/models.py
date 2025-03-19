#!/usr/bin/python3

from abc import ABC, abstractmethod
from os import environ
import base64
import numpy as np
import cv2
from PIL import Image
import io
import sys
import torch
import tempfile
import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

sys.path.append("PATH to DeepSeek-VL2")
from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor
from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class VQA(ABC):
  def encode_img(self, image):
    if isinstance(image, str):
      image = cv2.imread(image)
    else:
      assert isinstance(image, np.ndarray), "Input image must be a NumPy array"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_size = (1024, 1024)
    img_pil = Image.fromarray(image)
    img_pil.thumbnail(max_size)

    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='JPEG', quality=50)
    img_bytes = img_bytes.getvalue()

    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64
  @abstractmethod
  def inference(self, question, image, system = None):
    pass

class Qwen25VL7B_dashscope(VQA):
  def __init__(self, api_key):
    from openai import OpenAI
    self.client = OpenAI(
      api_key = api_key,
      base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
  def inference(self, question, image, system_message = None):
    img_data = self.encode_img(image)
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image_url', "image_url": {
        'url': f"data:image/png;base64,{img_data}"
      }}
    ]})
    response = self.client.chat.completions.create(
      model = 'qwen2.5-vl-7b-instruct',
      messages = messages,
    )
    return response.choices[0].message.content

class Qwen25VL7B_tgi(VQA):
  def __init__(self, tgi_host = "http://localhost:9091"):
    from huggingface_hub import InferenceClient
    self.client = InferenceClient(tgi_host)
  def inference(self, question, image, system_message = None):
    img_data = self.encode_img(image)
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image_url', "image_url": {
        'url': f"data:image/jpeg;base64,{img_data}"
      }}
    ]})
    response = self.client.chat_completion(
      messages = messages
    )
    return response.choices[0].message.content

class Qwen25VL7B_transformers(VQA):
  def __init__(self, huggingface_api_key, device = 'cuda'):
    assert device in {'cuda', 'cpu'}
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype = "auto", device_map = "auto", low_cpu_mem_usage = True if device == 'cpu' else None
    ).to(device)
    self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
  def inference(self, question, image, system_message = None):
    from qwen_vl_utils import process_vision_info
    img_data = self.encode_img(image)
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image', "image": f"data:image/png;base64,{img_data}"}
    ]})
    text = self.processor.apply_chat_template(
      messages, tokenize = False, add_generation_prompt = True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
      text = [text],
      images = image_inputs,
      videos = video_inputs,
      padding = True,
      return_tensors = "pt"
    )
    inputs = inputs.to(next(self.model.parameters())[0].device)
    generated_ids = self.model.generate(**inputs)
    generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self.processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens = True, clean_up_tokenization_spaces = False
    )
    return output_text[0]

class DeepSeek_VL2_transformers(VQA):
  def __init__(self, huggingface_api_key, device='cuda'):
    assert device in {'cuda', 'cpu'}
    environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
    model_path = 'deepseek-ai/deepseek-vl2-tiny'
    self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    self.tokenizer = self.vl_chat_processor.tokenizer
    self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

  def inference(self, question, image, system_message=None):
    if isinstance(image, np.ndarray):
      image = Image.fromarray(image)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
      image.save(temp_file.name, format="JPEG")
      image_path = temp_file.name

    if system_message is not None:
      question = system_message + '\n\n' + question

    conversation = [
      {
        "role": "<|User|>",
        "content": f"{question}\n <image>",
        "images": [image_path],
      },
      {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    if not pil_images:
      raise ValueError("Error: Image loading failed! Check `load_pil_images()`.")

    print(f"Loaded {len(pil_images)} images successfully.")

    prepare_inputs = self.vl_chat_processor(
      conversations=conversation,
      images=pil_images,
      force_batchify=True,
      system_prompt=""
    ).to(self.vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    if inputs_embeds is None or inputs_embeds.nelement() == 0:
      raise ValueError("Error: inputs_embeds is empty! Check image processing.")

    print(f"Debug: inputs_embeds shape: {inputs_embeds.shape}")

    outputs = self.vl_gpt.generate(
      inputs_embeds=inputs_embeds,
      input_ids=prepare_inputs.input_ids,
      images=prepare_inputs.images,
      images_seq_mask=prepare_inputs.images_seq_mask,
      images_spatial_crop=prepare_inputs.images_spatial_crop,
      attention_mask=prepare_inputs.attention_mask,
      # past_key_values=past_key_values,

      pad_token_id=self.tokenizer.eos_token_id,
      bos_token_id=self.tokenizer.bos_token_id,
      eos_token_id=self.tokenizer.eos_token_id,
      max_new_tokens=512,

      do_sample=False,
      use_cache=True,
    )


    answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    structured_response = f"{prepare_inputs['sft_format'][0]} {answer}"
    return structured_response
