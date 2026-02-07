import pandas as pd
import numpy as np
import joblib
import json
from itertools import chain
import click
import os
import glob

from omegaconf import OmegaConf

import sys
sys.path.append('.')

@click.command()
def main():
    
    conf = OmegaConf.load('params.yaml')
    mitre_attack_df = pd.read_csv(conf['get_data']['data_mitre_attack_proc_fn'])
    mitre_attack_df = mitre_attack_df.assign(labels = mitre_attack_df['labels'].map(lambda x:[x]))

    
    with open(conf['get_data']['label2tactic_fn'], 'rt') as f_r:
        label2tactic = json.load(f_r)

    tram_df = pd.read_json(conf['get_data']['tram_fn']).drop(columns='doc_title')
    sel = tram_df.sentence.str.findall(';').str.len()>0

    tram_df = tram_df[~sel]
    tram_df = tram_df.drop_duplicates(subset='sentence')
    
    mitr_df = mitre_attack_df

    if conf['get_data']['use_tram_f']:
        df = pd.concat([mitr_df, tram_df], ignore_index=True)
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
    
    df['origin_ttp'] = df['labels']
    if conf['get_data']['ignore_subt']:
        df['ttp'] = df['labels'].map(lambda x: [it.split('.')[0] for it in x])

    df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] if it in label2tactic else '' for it in x ])))

    df.sentence = df.sentence.str.strip()

    # еще есть скрытые, у которых отличия только в паре слов или символов
    df['shadow_duples'] = df.sentence.str[:40]+ df.sentence.str[50] + df.sentence.str[-40:]

    df = df.drop_duplicates('shadow_duples').drop(columns='shadow_duples')

    
    # при разбиении на абзацы описаний вылезают дубли
    df = df.drop_duplicates('sentence')

    # есть тексты очень малые и при разбиении на абзацы списки превращаются в мини перечисления
    df = df[df['sentence'].str.split().str.len()>=5].reset_index(drop=True)
    

    df.to_csv(conf['get_data']['data_fn'], index=False, escapechar='\\')
    
if __name__=='__main__':

    main()