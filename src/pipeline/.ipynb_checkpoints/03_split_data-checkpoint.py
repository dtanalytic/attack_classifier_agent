import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import click
from omegaconf import OmegaConf

import sys
sys.path.append('.')
from src.constants import ActivityLogger

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

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__) 
        conf = OmegaConf.load('params.yaml')
        
        SEED = conf['seed']
        np.random.seed(SEED)
        
        data = pd.read_csv(conf['get_data']['data_fn'], escapechar='\\')
        data['labels'] = data['labels'].map(lambda x: eval(x))
        data['origin_ttp'] = data['origin_ttp'].map(lambda x: eval(x))
        data['ttp'] = data['ttp'].map(lambda x: eval(x))
        data['sentence'] = data['sentence'].str.replace(r'\s+', ' ', regex=True)
        
        if conf['get_data']['replace_entities']:
            # начиная с python 3.7 порядок ключей сохраняется, поэтому можно не упорядочивать
            # pat_d = {it:globals()[it] for it in globals() if 'regexp_' in it and not '_upd' in it}
            # data['sentence'] = data['sentence'].map(lambda x: replace_entities(x, pat_d))
            
            pat_d_upd = {it:globals()[it] for it in globals() if 'regexp_' in it and '_upd' in it}
            data['sentence'] = data['sentence'].map(lambda x: replace_entities_upd(x, pat_d_upd))

        # другие названия в квадратных скобках убираем
        data['sentence'] = data['sentence'].str.replace(r'\[(\w+)\]', r'\1', regex=True)
        
        # после того, как убрал названия в круглых скобках вылезли дубли, например, в отчете та же формулировка, как
        # и в первоисточнике на митр только уже без круглых скобок
        data = data.drop_duplicates(subset=['sentence']).reset_index(drop=True)
        
        # юникод, накоторых токенизатор fastai жаловался
        symb_l = ['\xe4', '\u202f', '\u2192']
        for symb in symb_l:
            data['sentence'] = data['sentence'].str.replace(symb, '')

        ttp_l = data['ttp'].explode('ttp').loc[lambda x: x.notna()].unique().tolist()

        mlb_ttp = MultiLabelBinarizer()
        mlb_ttp.fit([[c] for c in ttp_l])
        
        
        joblib.dump(mlb_ttp, conf['get_data']['ttp_mlb_fn'])
        
        data['enc_ttp'] = mlb_ttp.transform(data['ttp']).tolist()

        if not conf['val_nmitr']:
            val_ts_size = conf['val_ts_size']

            split_col = 'enc_ttp'
            
            mskf = MultilabelStratifiedKFold(n_splits=int(1/(2*val_ts_size)), shuffle=True, random_state=SEED)
            # позиции от 0 до n
            for tr_idx, val_ts_idx in mskf.split(data.values, np.array(data[split_col].tolist())):
                break
            
            mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
            
            # позиции от 0 до m
            for val_idx, ts_idx in mskf.split(data.iloc[val_ts_idx].values, np.array(data[split_col].iloc[val_ts_idx].tolist())):
                break
            
            val_idx = val_ts_idx[val_idx]
            ts_idx = val_ts_idx[ts_idx]
            
            data['split'] = 'tr'
    
            data.loc[data.index[val_idx], 'split'] = 'val'
            data.loc[data.index[ts_idx], 'split'] = 'ts'
        
        else:
        
            mitre_sel = data['url'].str.contains('https://attack.mitre.org', na=False)
            other_nempty_sel = (~mitre_sel) & (data['ttp'].map(lambda x: len(x)>0))
            other_empty_sel = (~mitre_sel) & (data['ttp'].map(lambda x: len(x)==0))
            
            tr_idx = data[mitre_sel].index
            val_idx = data[~mitre_sel].index
            data['split'] = 'val'
        
            # пустых в пропорции столько, сколько непустых
            frac = mitre_sel.sum()/(mitre_sel.sum()+other_nempty_sel.sum())
            
            N = other_empty_sel.sum()
            seq_idx_empty_tr = set(np.random.choice(range(N), size=int(N*frac), replace=False))
            seq_idx_empty_val = set(range(N)) - seq_idx_empty_tr
            
            empty_idx = data.loc[other_empty_sel].index
            
            empty_val_idx = empty_idx.values[list(seq_idx_empty_val)].tolist()
            tr_mod_idx = tr_idx.tolist() + empty_idx.values[list(seq_idx_empty_tr)].tolist()
            
            data.loc[tr_mod_idx, 'split'] = 'tr'
        
            idx_ts = data[data.split=='val'].sample(frac=0.5).index
            
            data.loc[idx_ts, 'split']='ts'
        
        data.to_csv(conf['get_data']['split_fn'], index=False)
        
        logger.info(f"Получившееся разделение для больших классов - {data.explode('origin_ttp').groupby(['split', 'origin_ttp']).size().loc[lambda x: x>100]}")
        

        ActivityLogger().close_logger(logger)
    
    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)
    
if __name__=='__main__':

    main()