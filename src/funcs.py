import re
import random
import numpy as np
from omegaconf import OmegaConf

import pandas as pd

import torch

import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

import string
from string import punctuation

from pymystem3 import Mystem

import requests

from fastcore.basics import store_attr

from typing import Any, List, Dict, Mapping, Optional


from langgraph.graph.message import MessageGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

conf_seed = OmegaConf.load('params.yaml')
set_seed(conf_seed['seed'])


def preprocess_text(text, to_lower=True, min_len_token=3, max_len_token=10, stop_words_l=None, stem_lemm_flag=None, save_text=True, save_only_words=True, lang='russian'):

    if save_only_words:
        text = re.sub('(?u)[^\w\s]', '', text)
    text = re.sub(' +', ' ', text)

    if to_lower:
        text = text.lower()

    if stem_lemm_flag=='lemm':
        text = ''.join(Mystem().lemmatize(text))
        stem_lemm_flag=None

    words_l = nltk.tokenize.word_tokenize(text)
    words_l = preprocess_words(words_l, to_lower=False, min_len_token=min_len_token, max_len_token = max_len_token, stop_words_l=stop_words_l, 
                               stem_lemm_flag=stem_lemm_flag, save_only_words=False, lang=lang)

    if save_text:
        return ' '.join(words_l)
    else:
        return words_l

def preprocess_words(words_l, to_lower=True, min_len_token=3, max_len_token=10, stop_words_l=None, 
                     stem_lemm_flag=None, save_only_words=True, lang='russian'):

    if save_only_words:
        punct=list(punctuation)
        words_l = [w for w in words_l if not w.isdigit() and not w in punct]

    if min_len_token:
        words_l = [w for w in words_l if len(w)>=min_len_token]
        
    if max_len_token:
        words_l = [w for w in words_l if len(w)<=max_len_token]

    if stem_lemm_flag=='stem':

        stemmer = SnowballStemmer(lang)
        words_l = [stemmer.stem(w) for w in words_l]

    elif stem_lemm_flag=='lemm':

        lemm = Mystem()
        words_l = [lemm.lemmatize(w) for w in words_l]

    if stop_words_l:
        words_l = [w for w in words_l if not w in stop_words_l]

    if to_lower:
        words_l = [w.lower() for w in words_l]

    return words_l


class Vikhr(BaseChatModel):

      max_tokens: int = 3000
      temperature: float = 0
      model_name: str = "Vikr"

      def _generate(
          self,
          messages: List[BaseMessage],
          stop: Optional[List[str]] = None,
          run_manager: Optional[CallbackManagerForLLMRun] = None,
          **kwargs: Any,
      ) -> ChatResult:

            formatted_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    formatted_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": message.content})
                else:
                    formatted_messages.append({"role": "user", "content": message.content})
                    
            response = send_post_model(formatted_messages, model_nm="Vikhr12BInstruct", temperature=self.temperature, max_tokens=self.max_tokens)
            
            
            message = AIMessage(content=response['choices'][0]['message']['content'])
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

      @property
      def _llm_type(self) -> str:
          return "Vikr"

      @property
      def _identifying_params(self) -> Mapping[str, Any]:
          """Get the identifying parameters."""
          return {"max_token": self.max_tokens, "model_name": self.model_name,
                  "temperature": self.temperature}




def get_rag_db(conf):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embed_wrapper = HuggingFaceEmbeddings(model_name=conf['storage']['smodel_path'],
                                               model_kwargs={'device': device})
    
    return FAISS.load_local(conf['storage']['db_path'], embed_wrapper, allow_dangerous_deserialization=True)
    
def form_rag_query(query, prefix, rag_db, search_kwargs):

    if 'max_marginal' == search_kwargs['method']:
        assert all([it in search_kwargs.keys() for it in ['fetch_k', 'k', 'lambda_mult']]), 'with max_marginal method all must be in search_kwargs fetch_k, k, lambda_mult'
        docs = rag_db.max_marginal_relevance_search(query, fetch_k = search_kwargs['fetch_k'],
                                                k=search_kwargs['k'], lambda_mult=search_kwargs['lambda_mult'])
    l = [f'Contents:"{doc.page_content}", Techniques - {doc.metadata["techniques"]}, Tactics - {doc.metadata["tactics"]}, MITRE root title - {doc.metadata["name"]}, MITRE url - {doc.metadata["url"]}'
      if doc.metadata["techniques"]!='' else f'Contents:"{doc.page_content}", no malicious content found' for _, doc in enumerate(docs)]
    s = '\n'.join(l)
    rag_s = f'''Below will be texts only for rag purposes, don't analyse them, only keep in mind to better classify text after key phrase "Text to analyse":\n{s}'''

    fin_query = prefix+f"\n{rag_s}\n\n\"{query}\""

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

    return response, response['choices'][0]['message']['content']



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



