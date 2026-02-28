import pandas as pd
import numpy as np
import joblib
import time
import click

import json

from omegaconf import OmegaConf
from dotenv import load_dotenv


from sklearn.metrics import (log_loss, roc_auc_score, average_precision_score, f1_score, 
                            precision_recall_fscore_support, confusion_matrix)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever

from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.output_parsers import StrOutputParser

load_dotenv('cred.env')

def get_answer(x, qa_chain, parser_instructions, pred_bert=False):

    try:
        if pred_bert:
            result = qa_chain.invoke({'parser_instructions':parser_instructions, 'input':x['sentence'], 
                                      'model_preds':', '.join(eval(x['pred_bert']))})
        else:
            result = qa_chain.invoke({'parser_instructions':parser_instructions, 'input':x['sentence']})            
        return result['answer']
    except Exception as e:
        return [{'error':e}]
        

import sys
sys.path.append('.')

from src.constants import ActivityLogger, TTPList
from src.constants import system_template, user_template, document_prompt, user_bert_template
from src.funcs import get_rag_db


def predict(pred_l):
    '''
    Args:
    ----------
    pred_l - list of strings
    Returns
    -------
    pandas DataFrame 
    
    '''
    
    conf = OmegaConf.load('params.yaml')

    np.random.seed(conf['seed'])

    pred_df = pd.DataFrame({'sentence':pred_l})
    
    parser = PydanticOutputParser(pydantic_object=TTPList)


    if conf['train_eval']['use_bert']:
        from src.constants import user_bert_template
        prompt = ChatPromptTemplate.from_messages(
                                                 [("system", system_template),
                                                  ("user", user_bert_template)])
    else:
        prompt = ChatPromptTemplate.from_messages(
                                             [("system", system_template),
                                              ("user", user_template)])
    

    if conf['train_eval']['model']=='vikhr':
        from src.funcs import Vikhr
        llm_model = Vikhr()
        
    elif conf['train_eval']['model']=='gpt':
        from langchain_openai import ChatOpenAI
        import httpx    
        
        http_client = httpx.Client(proxy="http://5.129.232.80:4080", timeout=10.0)   
        llm_model = ChatOpenAI(api_key=os.environ['API_KEY'], 
                        http_client=http_client, model="gpt-5-mini", temperature=0.0)
        
    
    db = get_rag_db(conf, key='prod')

    doc_chain = create_stuff_documents_chain(
        llm=llm_model,
        prompt=prompt,
        document_variable_name="context",
        document_prompt=document_prompt,
        document_separator='\n-------------\n',
        output_parser=parser 
    )

    if conf['train_eval']['rag_type']=='embed':
        k = eval(conf['train_eval']['rag_k'])[0]
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k, 'lambda_mult': conf['train_eval']['lambda_mult'],
                                                                 'fetch_k': conf['train_eval']['fetch_k']})
    elif conf['train_eval']['rag_type']=='word':
        k = eval(conf['train_eval']['rag_k'])[0]
        retriever = joblib.load(conf['prod']['bm25_path'])
        retriever.k = k
    else:
        ks = eval(conf['train_eval']['rag_k'])
        assert len(ks)==2, 'input 2 values of k for retrievers'
            
        vector_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": ks[0], 'lambda_mult': conf['train_eval']['lambda_mult'],
                                                                 'fetch_k': conf['train_eval']['fetch_k']})
        bm_retriever = joblib.load(conf['prod']['bm25_path'])
        bm_retriever.k = ks[1]
        
        retriever = EnsembleRetriever(
            retrievers=[bm_retriever, vector_retriever],
            weights=[1.0, 1.0]
        )
    
    qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

    
    mlb_ttp = joblib.load(conf['prod']['ttp_mlb_fn'])        
    # pred_df['resp'] = pred_df['sentence'].map(lambda x: get_answer(x, qa_chain, parser))

    # debug part
    parser_deb = StrOutputParser()
    llm_model_deb = ParrotFakeChatModel()
    
    doc_chain_deb = create_stuff_documents_chain(
        llm=llm_model_deb,
        prompt=prompt,
        document_variable_name="context",
        document_prompt=document_prompt,
        document_separator='\n-------------\n',
        output_parser=parser_deb 
    )
    
    qa_chain_deb = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain_deb)
    
    pred_df['query'] = pred_df.apply(lambda x: get_answer(x, qa_chain_deb, parser_instructions='parser_instructions', pred_bert=conf['train_eval']['use_bert']), axis=1)    
    pred_df['resp'] = pred_df.apply(lambda x: get_answer(x, qa_chain, parser.get_format_instructions(), pred_bert=conf['train_eval']['use_bert']), axis=1)


    # остальные error
    pred_df['pred_ttp'] = pred_df['resp'].map(lambda x: [it.mitre_id.split('.')[0] for it in x.ttps] if isinstance(x, TTPList) else [])
    pred_df['pred_ttp'] = pred_df['pred_ttp'].map(lambda x: list(set(x)))
    pred_df['pred_enc_ttp'] = pred_df['pred_ttp'].map(lambda x: mlb_ttp.transform([x])[0].tolist())

    return pred_df


