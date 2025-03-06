#!/usr/bin/python3

from os import environ
import json
import json_repair
import random
import string
from langchain import hub
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, ToolMessage
from configs import *

class ChatHuggingFace2(ChatHuggingFace):
  def generate_random_sequence(self, length = 24):
    characters = string.ascii_letters + string.digits
    random_sequence = ''.join(random.choice(characters) for _ in range(length))
    return random_sequence
  def _generate(
    self,
    messages,
    stop = None,
    run_manager = None,
    **kwargs,
  ):
    # ordinary LLM inference
    llm_input = self._to_chat_prompt(messages, **kwargs)
    llm_result = self.llm._generate(
      prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
    )
    generations = self._to_chat_result(llm_result)
    # NOTE: if the query is about tool calling, parse the result
    if 'tools' in kwargs:
      try:
        tool_calls = json.loads(generations.generations[0].message.content)
      except:
        try:
          tool_calls = json_repair.loads(generations.generations[0].message.content)
        except:
          raise
      if type(tool_calls) is dict:
        tool_calls = [tool_calls]
      generations.generations[0].message.content = ''
      for call in tool_calls:
        generations.generations[0].message.tool_calls.append({
          'name': call['name'],
          'args': call['parameters'],
          'id': f'call_{self.generate_random_sequence()}'
        })
    return generations
  def _to_chat_prompt(
    self,
    messages,
    **kwargs,
  ) -> str:
    """Convert a list of messages into a prompt format expected by wrapped LLM."""
    if not messages:
      raise ValueError("At least one HumanMessage must be provided!")
    if not isinstance(messages[-1], HumanMessage):
      raise ValueError("Last message must be a HumanMessage!")
    messages_dicts = [self._to_chatml_format(m) for m in messages]
    # NOTE: add binded kwargs to tokenizer
    return self.tokenizer.apply_chat_template(
      messages_dicts, tokenize=False, add_generation_prompt=True, **kwargs
    )

class Llama3_2(ChatHuggingFace2):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token
    super(ChatHuggingFace, self).__init__(
      llm = HuggingFaceTextGenInference(
        inference_server_url = tgi_host,
        do_sample = False,
        top_p = 0.8,
        temperature = 0.8
      ),
      tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'),
      verbose = True
    )

class Qwen2_5(ChatHuggingFace2):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token
    super(ChatHuggingFace, self).__init__(
      llm = HuggingFaceTextGenInference(
        inference_server_url = tgi_host,
        do_sample = False,
        top_p = 0.8,
        temperature = 0.8,
      ),
      tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'),
      verbose = True
    )

if __name__ == "__main__":
  from langchain_core.tools import tool
  from langchain.agents import AgentExecutor, create_tool_calling_agent

  @tool
  def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

  @tool
  def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

  chat_model = Llama3_2()
  tools = [add, multiply]
  if False:
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": "What is 3 * 12?"})
  else:
    chat_model = chat_model.bind_tools(tools)
    response = chat_model.invoke([('user', 'What is 3 * 12?')])
    print(response.tool_calls)
  print(response)
