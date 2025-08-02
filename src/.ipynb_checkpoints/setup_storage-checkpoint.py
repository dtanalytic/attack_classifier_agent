import pandas as pd
import numpy as np
import joblib
import time
import click

from ruamel.yaml import YAML

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import DistanceStrategy

from langchain_huggingface import HuggingFaceEmbeddings
import torch

import sys
sys.path.append('.')
from src.constants import ActivityLogger

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        
        conf = YAML().load(open('params.yaml'))
        base_df = pd.read_csv(conf['get_data']['data_fn'])
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        embed_wrapper = HuggingFaceEmbeddings(model_name=conf['storage']['smodel_path'],
                                           model_kwargs={'device': device})
        
        
        sents = base_df['sentence'].tolist()
        
        metas = [{'techniques':', '.join(eval(row.origin_ttp)),'tactics': ', '.join(eval(row.labels)),
                  'url':row.url, 'name': row.par_name,
                  } for row in base_df.itertuples()]

        logger.info(f'Подготовили данные для загрузки в хранилище')
        
        if  conf['storage']['re_write_db']:
            start = time.time()
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