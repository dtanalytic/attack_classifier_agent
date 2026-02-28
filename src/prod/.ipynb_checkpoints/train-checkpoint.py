import pandas as pd
import numpy as np
import joblib

import os
import glob

from itertools import chain
from functools import partial

import click

import json
import re

from sklearn.preprocessing import MultiLabelBinarizer

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

import torch

from omegaconf import OmegaConf

import sys
sys.path.append('.')
from src.spec_funcs import load_mitr, prep_mitr

from src.constants import (regexp_email, regexp_cve, regexp_url, regexp_domain, regexp_registry,  regexp_fpath, 
                             regexp_fname, regexp_ipv4, regexp_ipv6,  regex_domain_zone,
                            regexp_hash_md5, regexp_hash_sha1, regexp_hash_sha256, regexp_hash_sha512, regexp_hash_ssdeep, 
                             regexp_coins_eth, regexp_coins_btc, regexp_coins_bch, regexp_coins_ltc,
                            regexp_coins_doge, regexp_coins_dash, regexp_coins_xmr, regexp_coins_neo, regexp_coins_xrp)

from src.constants import (regexp_email_upd, regexp_fname_upd, regexp_cve_upd, regexp_url_upd, regexp_domain_upd, regexp_registry_upd, 
                              regexp_ipv4_upd, regexp_ipv6_upd,  regexp_fpath_upd,  regex_domain_zone_upd,
                            regexp_hash_md5_upd, regexp_hash_sha1_upd, regexp_hash_sha256_upd, regexp_hash_sha512_upd, regexp_hash_ssdeep_upd,
                            regexp_coins_eth_upd, regexp_coins_btc_upd, regexp_coins_bch_upd, regexp_coins_ltc_upd,
                            regexp_coins_doge_upd, regexp_coins_dash_upd, regexp_coins_xmr_upd, regexp_coins_neo_upd, regexp_coins_xrp_upd)

from src.spec_funcs import replace_entities, replace_entities_upd
from src.funcs import preprocess_text

def create_dbs():

    conf = OmegaConf.load('params.yaml')

    mitre_df, main_descr_df, proc_df = load_mitr(conf['get_data']['mitre_attack_fn'])
    mitr_df = prep_mitr(main_descr_df, proc_df, conf)

    print(f'Предобработали mitre json')

    label2tactic = mitr_df.set_index('labels')['kill_chain_tags'].to_dict()

    mitr_df = mitr_df.assign(labels = mitr_df['labels'].map(lambda x:[x]))


    tram_df = pd.read_json(conf['get_data']['tram_fn']).drop(columns='doc_title')
    sel = tram_df.sentence.str.findall(';').str.len()>0

    tram_df = tram_df[~sel]
    tram_df = tram_df.drop_duplicates(subset='sentence')
    

    if conf['get_data']['use_tram_f']:
        df = pd.concat([mitr_df, tram_df], ignore_index=True)
        print(f'Добавили tram')
    else:
        df = mitr_df

    
    
    if conf['get_data']['add_cti_hal']:
        
        fns = glob.glob(f"{conf['get_data']['cti_hal_dn']}/**/*.json", recursive=True)
        
        hal_df = pd.concat([pd.read_json(it)[['context', 'technique']].rename(columns={'context':'sentence', 'technique':'labels'})\
                                .assign(url=f'https://github.com/dessertlab/CTI-HAL/data{it.split("data/external/cti_hal")[1]}'.replace('\\', '/'))
                            for it in fns], ignore_index=True)
    
        hal_df = hal_df[hal_df['labels'].notna()]
        hal_df.labels=hal_df.labels.map(lambda x: [x])
        hal_df = hal_df[~hal_df.duplicated(subset='sentence')]
        df = pd.concat([df, hal_df], ignore_index=True)
        
        print(f'Добавили cti_hal')
        
    if conf['get_data']['use_reports_f']:
        DN = conf['get_data']['rep_dn']
        fns = [f'{DN}/{it}' for it in os.listdir(DN) if 'json' in it]
        
        tab_df = pd.concat([pd.read_json(fn).explode('tables') for fn in fns], ignore_index=True)
        tab_df['tables'] = tab_df['tables'].map(lambda x: [(it1, it2) 
                                        for it1, it2 in zip(x['TechniqueID'].values(), x['Procedure'].values())])
        tab_df = tab_df.explode('tables').drop_duplicates().reset_index(drop=True)
        tab_df['technic'] = tab_df['tables'].map(lambda x: x[0])
        tab_df['sentence'] = tab_df['tables'].map(lambda x: x[1])
        
        tab_df = tab_df[tab_df['sentence'].str.split().str.len()>3]
        tab_df['labels'] = tab_df['technic'].map(lambda x: [x.upper()])
        
        # тут и мобильные угрозы есть, поэтому фильтруем их
        tab_df = tab_df[tab_df['labels'].map(lambda x: all([it in label2tactic for it in x]))]

        df = pd.concat([df, tab_df[['sentence',	'labels', 'report_path']].rename(columns={'report_path':'url'})], ignore_index=True)

        print(f'Добавили отчеты')
        
    df['origin_ttp'] = df['labels']
    if conf['get_data']['ignore_subt']:
        df['ttp'] = df['labels'].map(lambda x: [it.split('.')[0] for it in x])

    df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] if it in label2tactic else '' for it in x ])))

    df.sentence = df.sentence.str.strip()

    # еще есть скрытые, у которых отличия только в паре слов или символов
    df['shadow_duples'] = df.sentence.str[:40]+ df.sentence.str[50] + df.sentence.str[-40:]

    df = df.drop_duplicates('shadow_duples').drop(columns='shadow_duples')
    df = df.drop_duplicates('sentence')

    # есть тексты очень малые и при разбиении на абзацы списки превращаются в мини перечисления
    data = df[df['sentence'].str.split().str.len()>=5].reset_index(drop=True).copy()
    del df
    
    data['sentence'] = data['sentence'].str.replace(r'\s+', ' ', regex=True)
    
    if conf['get_data']['replace_entities']:
        # начиная с python 3.7 порядок ключей сохраняется, поэтому можно не упорядочивать
        # pat_d = {it:globals()[it] for it in globals() if 'regexp_' in it and not '_upd' in it}
        # data['sentence'] = data['sentence'].map(lambda x: replace_entities(x, pat_d))
        
        pat_d_upd = {it:globals()[it] for it in globals() if 'regexp_' in it and '_upd' in it}
        data['sentence'] = data['sentence'].map(lambda x: replace_entities_upd(x, pat_d_upd))

    print(f'Применили маскирование')
    # другие названия в квадратных скобках убираем
    data['sentence'] = data['sentence'].str.replace(r'\[(\w+)\]', r'\1', regex=True)
    
    # после того, как убрал названия в круглых скобках вылезли дубли, например, в отчете та же формулировка, как
    # и в первоисточнике на митр только уже без круглых скобок
    data = data.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    
    # юникод, на которых токенизатор fastai жаловался
    symb_l = ['\xe4', '\u202f', '\u2192']
    for symb in symb_l:
        data['sentence'] = data['sentence'].str.replace(symb, '')

    ttp_l = data['ttp'].explode('ttp').loc[lambda x: x.notna()].unique().tolist()

    mlb_ttp = MultiLabelBinarizer()
    mlb_ttp.fit([[c] for c in ttp_l])
    joblib.dump(mlb_ttp, conf['prod']['ttp_mlb_fn'])
    
    data['enc_ttp'] = mlb_ttp.transform(data['ttp']).tolist()

    print(f'Создали кодировщик ttp и закодировали label-ы')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embed_wrapper = HuggingFaceEmbeddings(model_name=conf['prod']['smodel_path'],
                                       model_kwargs={'device': device})
    
    
    sents = data['sentence'].tolist()
    
    metas = [{'techniques':', '.join(row.origin_ttp),'tactics': ', '.join(row.labels),
              'url':row.url, 'name': row.name,
              } for row in data.itertuples()]


    prep_func = partial(preprocess_text, to_lower=False, save_text=False, stop_words_l=False, 
                                            save_only_words=False, lang='english',  min_len_token=3, max_len_token=10, stem_lemm_flag='stem')
    
    bm_retriever = BM25Retriever.from_texts(texts=sents, metadatas=metas, k=2, ids=range(len(sents)),
                                            preprocess_func=prep_func)
    
    joblib.dump(bm_retriever, conf['prod']['bm25_path'])
    
    
    db = FAISS.from_texts(sents, embed_wrapper, metadatas=metas, ids=range(len(sents)),
                  distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True)
    
    db.save_local(conf['prod']['db_path'])

    print(f'Создали хранилище и bm25')
