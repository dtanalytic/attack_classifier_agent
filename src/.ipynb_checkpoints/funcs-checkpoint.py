import re
import random
import numpy as np
from ruamel.yaml import YAML

import pandas as pd


import torch


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


