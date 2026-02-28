import pandas as pd
import numpy as np
import joblib
import time
import click

from functools import partial
from omegaconf import OmegaConf

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import torch

import sys
sys.path.append('.')
from src.constants import ActivityLogger
from src.funcs import preprocess_text

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        
        conf = OmegaConf.load('params.yaml')
        base_df = pd.read_csv(conf['get_data']['split_fn']).query('split=="tr"')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        embed_wrapper = HuggingFaceEmbeddings(model_name=conf['storage']['smodel_path'],
                                           model_kwargs={'device': device})
        
        
        sents = base_df['sentence'].tolist()
        

        metas = [{'techniques':', '.join(eval(row.origin_ttp)) if len(eval(row.origin_ttp)) >0 else np.nan,'tactics': ', '.join(eval(row.labels)) if len(eval(row.labels)) >0 else np.nan,
          'url':row.url, 'name': row.name,
          } for row in base_df.itertuples()]
        
        logger.info(f'Подготовили данные для загрузки в хранилище')
        
        if  conf['storage']['re_write_db']:
            start = time.time()

            prep_func = partial(preprocess_text, to_lower=False, save_text=False, stop_words_l=False, 
                                                    save_only_words=False, lang='english',  min_len_token=3, max_len_token=10, stem_lemm_flag='stem')
            
            bm_retriever = BM25Retriever.from_texts(texts=sents, metadatas=metas, k=2, ids=range(len(sents)),
                                                    preprocess_func=prep_func)
            
            joblib.dump(bm_retriever, conf['storage']['bm25_path'])
            
            if conf['storage']['db_type']=='faiss':
                
                db = FAISS.from_texts(sents, embed_wrapper, metadatas=metas, ids=range(len(sents)),
                              distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True)
                
                db.save_local(conf['storage']['db_path'])

            
            end = time.time()
            
            logger.info(f"Загрузка в хранилище {conf['storage']['db_type']} заняла {(end-start)/60:.1f} минут")
        else:
            logger.info(f"Внимание хранилище не перезаписывается")
        

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)
        
if __name__=='__main__':

    main()