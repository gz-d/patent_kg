#!/usr/bin/python3

from typing import Any, Union, List, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate, PromptTemplate
from langchain_experimental.graph_transformers.llm import create_unstructured_prompt
from configs import node_types, examples

def extract_triplets_template(node_types: Optional[List[str]] = None,
                              rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None):
  import langchain_experimental
  assert langchain_experimental.__version__ >= '0.3.3'
  chat_prompt = create_unstructured_prompt(node_types, rel_types, relationship_type = 'tuple')
  chat_prompt = ChatPromptTemplate(chat_prompt.messages)
  return chat_prompt

