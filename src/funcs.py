import re
import random
import numpy as np
from ruamel.yaml import YAML

import pandas as pd


import torch

import requests

from langgraph.graph.message import MessageGraph
from langgraph.checkpoint.memory import MemorySaver
from fastcore.basics import store_attr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

conf_seed = YAML().load(open('params.yaml'))
set_seed(conf_seed['seed'])


def get_rag_db(conf):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embed_wrapper = HuggingFaceEmbeddings(model_name=conf['storage']['smodel_path'],
                                               model_kwargs={'device': device})
    
    return FAISS.load_local(conf['storage']['db_path'], embed_wrapper, allow_dangerous_deserialization=True)
    
def form_rag_query(query, rag_db, search_kwargs):

    if 'max_marginal' == search_kwargs['method']:
        assert all([it in search_kwargs.keys() for it in ['fetch_k', 'k', 'lambda_mult']]), 'with max_marginal method all must be in search_kwargs fetch_k, k, lambda_mult'
        docs = rag_db.max_marginal_relevance_search(query, fetch_k = search_kwargs['fetch_k'],
                                                k=search_kwargs['k'], lambda_mult=search_kwargs['lambda_mult'])
    l = [f'Contents:"{doc.page_content}", Techniques - {doc.metadata["techniques"]}, Tactics - {doc.metadata["tactics"]}, MITRE root title - {doc.metadata["name"]}, MITRE url - {doc.metadata["url"]}'
      if doc.metadata["techniques"]!='' else f'Contents:"{doc.page_content}", no malicious content found' for _, doc in enumerate(docs)]
    s = '\n'.join(l)
    rag_s = f'''List of similar texts are below, among them may be texts with no malicious content, keep this in mind while classifying the given text:\n{s}'''

    fin_query = f"Classify given query based on similar texts below, last sentence form like:Output: TTP CODE, example: OUTPUT: T1585:\nQUERY:\"{query}\",\n{rag_s}"

    return fin_query



def send_post_model(messages, model_nm="DSR1DistallQwen32B", temperature=0, max_tokens=1000, token="token"):

    if model_nm=="DSR1DistallQwen32B":
        url = 'http://10.100.0.31:9092/v1/chat/completions'
    elif model_nm=="Vikhr12BInstruct":
        url = 'http://10.100.0.31:9091/v1/chat/completions'

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": model_nm,
      "messages": messages,
      "max_tokens": max_tokens,
      "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def ask_nomemory(text, model_nm="DSR1DistallQwen32B", temperature=0, max_tokens=1000, token="token"):

    messages = [
        {
          "role": "user",
          "content": text
        }
      ]
    response = send_post_model(messages, model_nm=model_nm, temperature=temperature, max_tokens=max_tokens, token=token)

    return response['choices'][0]['message']['content']



class ChatLLM():

    def __init__(self, model_nm, session_id, temperature, max_tokens, token='token'):
        store_attr()

    def convert_messages(self, message_list):
        converted = []
        for msg in message_list:
            # Определяем роль на основе типа сообщения
            if msg.__class__.__name__ == 'HumanMessage':
                role = "user"
            elif msg.__class__.__name__ == 'AIMessage':
                role = "assistant"  
            elif msg.__class__.__name__ == 'SystemMessage':
                role = "system"
            else:
                continue  # Пропускаем неизвестные типы сообщений
            
            # Добавляем в результирующий список
            converted.append({
                "role": role,
                "content": msg.content
            })
        return converted


    def chat_llm(self, messages):

        response = send_post_model(self.convert_messages(messages), model_nm=self.model_nm, temperature=self.temperature, max_tokens=self.max_tokens, token=self.token)
        return [AIMessage(response['choices'][0]['message']['content'])]
        
    def compile(self):
        builder = MessageGraph()
        builder.add_node("chat_llm", self.chat_llm)
        builder.set_entry_point("chat_llm")
        builder.set_finish_point("chat_llm")
        
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)

    def ask_memory(self, text):
        return self.convert_messages(self.graph.invoke([HumanMessage(text)], {"configurable": {"thread_id": self.session_id}}))



