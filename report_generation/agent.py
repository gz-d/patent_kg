#!/usr/bin/python3

from langchain.agents import load_tools, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_neo4j import Neo4jVector
from models import Llama3_2, Qwen2_5
from tools import load_chunk_retriever, load_document_retriever
from prompts import react_prompt

class Agent(object):
  def __init__(self, model = 'llama3', **kwargs):
    llms_types = {
      'llama3': Llama3_2,
      'qwen2': Qwen2_5
    }
    llm = llms_types[model]()
    vectordb = Neo4jVector(
      embedding = embedding,
      url = kwargs.get('host'),
      username = kwargs.get('username'),
      password = kwargs.get('password'),
      database = kwargs.get('db'),
      index_name = "typical_rag",
      search_type = "hybrid"
    )
    tools = [load_chunk_retriever(vectordb),
             load_document_retriever(vectordb)]
    prompt = react_prompt
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
