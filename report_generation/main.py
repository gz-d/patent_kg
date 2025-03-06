#!/usr/bin/python3

from absl import flags, app
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent import Agent
from configs import *

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', default = 'qwen2', enum_values = {'llama3', 'qwen2'}, help = 'model to use')

def create_interface():
  agent = Agent(model = FLAGS.model, host = neo4j_host, username = neo4j_user, password = neo4j_password, db = neo4j_db)
  def chatbot_response(user_input, history):
    chat_history = list()
    for human, ai in history:
      chat_history.append(HumanMessage(content = human))
      chat_history.append(AIMessage(content = ai))
    chat_history = chat_history[-max_history_len * 2:]
    response = agent.query(user_input, chat_history)
    history.append((user_input, response['output']))
    return "", history, history
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>QA System</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        user_input = gr.Textbox(label = '需要问什么？')
        with gr.Row():
          submit_btn = gr.Button("发送")
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot, state], value = "清空问题")
      submit_btn.click(chatbot_response,
                       inputs = [user_input, state],
                       outputs = [user_input, state, chatbot])
  return demo

def main(unused_argv):
  demo = create_interface()
  demo.launch(server_name = service_host,
              server_port = service_port)

if __name__ == "__main__":
  add_options()
  app.run(main)
