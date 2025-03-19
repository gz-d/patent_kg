#!/usr/bin/python3

from langchain.agents import load_tools, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from models import Llama3_2, Qwen2_5, GPT35Turbo, GPT4O, Campus, Tongyi
from tools import load_chunk_retriever, load_document_retriever
from prompts import react_prompt

class Agent(object):
  def __init__(self, model, chunk_vectordb, chunk_store, doc_vectordb, doc_store):
    llms_types = {
      'llama3': Llama3_2,
      'qwen2': Qwen2_5,
      'gpt3.5': GPT35Turbo,
      'gpt4o': GPT4O,
      'campus': Campus,
      'tongyi': Tongyi
    }
    llm = llms_types[model]()
    tools = [load_chunk_retriever(chunk_vectordb, chunk_store),
             load_document_retriever(doc_vectordb, doc_store)]
    prompt = react_prompt
    # adapt prompt to openai's preference
    if model in ['gpt3.5', 'gpt4o']:
      for i in range(len(prompt)):
        if prompt[i][0] == 'user':
          prompt[i] = ('human', prompt[i][1])
    prompt = prompt.partial(
      tools = render_text_description(tools),
      tool_names = ", ".join([t.name for t in tools])
    )
    chain = {
      "input": lambda x: x["input"],
      "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
      "chat_history": lambda x: x["chat_history"]
    } | prompt | llm | ReActJsonSingleInputOutputParser()
    self.agent_chain = AgentExecutor(agent = chain, tools = tools, verbose = True, handle_parsing_errors = True)
  def query(self, question, chat_history):
    return self.agent_chain.invoke({"input": question, "chat_history": chat_history})