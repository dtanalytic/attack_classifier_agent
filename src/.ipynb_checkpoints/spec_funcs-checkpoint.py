import numpy as np
import pandas as pd
import joblib
import json

from itertools import chain
import re 
from ruamel.yaml import YAML


import sys
sys.path.append('.')

from src.funcs import set_seed
conf_seed = YAML().load(open('params.yaml'))
set_seed(conf_seed['seed'])


    
def load_mitr(fn):

    d=json.load(open(fn))
    attack_l = [it for it in d['objects'] if it['type']=='attack-pattern' if not 'revoked' in it.keys() or not it['revoked']]
    rel_l = [it for it in d['objects'] if it['type']=='relationship']
    subtech_l = [it for it in d['objects'] if it['type']=='relationship' and it['relationship_type']=='subtechnique-of']

    mitre_df = pd.DataFrame([(attack['id'], attack['name'], attack['description'], attack['external_references'][0]['external_id'],
                              attack['external_references'][0]['url'], attack['x_mitre_is_subtechnique'], attack['x_mitre_platforms'], attack['kill_chain_phases'],
                              attack['created'], attack['external_references'][1:],
                              attack['x_mitre_permissions_required'] if 'x_mitre_permissions_required' in attack.keys() else None,
                             attack['x_mitre_effective_permissions'] if 'x_mitre_effective_permissions' in attack.keys() else None,
                              attack['x_mitre_data_sources'] if 'x_mitre_data_sources' in attack.keys() else None,
                              attack['x_mitre_defense_bypassed'] if 'x_mitre_defense_bypassed' in attack.keys() else None
                              )
                              for attack in attack_l],
                            columns=['id', 'name', 'sentence', 'labels', 'url', 'subtechnique', 'platforms', 'kill_chain_phases',
                                     'created', 'doc_refs', 'permissions', 'effective_permissions', 'data_sources', 'defense_bypassed'])
    mitre_df['proc_flag'] = False
    
    mittre_l = []
    for _, row in mitre_df.iterrows():
    
        # proc_itms = [it for it in rel_l if (it['target_ref']==row['id']) and (it['relationship_type']=='uses')]
        proc_itms = [it for it in rel_l if (it['target_ref']==row['id']) and (it['relationship_type']=='uses') and (it['x_mitre_deprecated']==False if 'x_mitre_deprecated' in it else True)]

        det_itms = [it for it in rel_l if (it['target_ref']==row['id']) and (it['relationship_type']=='detects')]
        mit_itms = [it for it in rel_l if (it['target_ref']==row['id']) and (it['relationship_type']=='mitigates')]
    
    
        part_df = row.drop(['proc_flag']).to_frame().T
        part_df['proc_software_links'] = [list(chain(*[re.findall(r'\((https://[^\s\)]{1,})', it['description']) for it in proc_itms ]))]

        if part_df['subtechnique'].iloc[0]:
            part_df['par_id'] = [it for it in subtech_l if it['source_ref']==row['id']][0]['target_ref']
        else:
            part_df['par_id'] = None
            
        part_df['proc_descr_links'] = [[subit['url'] for it in proc_itms if 'external_references' in it for subit in it['external_references'] ]]
        part_df['proc_descr'] = [[it['description'] for it in proc_itms]]
        # break
    
        part_df['det_descr'] = [[it['description'] for it in det_itms if 'description' in it]]
        part_df['det_data_comp'] = [[it['source_ref'] for it in det_itms if 'source_ref' in it]]
        part_df['mit_descr'] = [[it['description'] for it in mit_itms if 'description' in it]]
        part_df['mit_coa'] = [[it['source_ref'] for it in mit_itms if 'source_ref' in it]]
    
        mittre_l.append(part_df)
    
    mitre_df = pd.concat(mittre_l, ignore_index=True)
    mitre_df['kill_chain_tags'] = mitre_df['kill_chain_phases'].map(lambda x: [it['phase_name'] for it in x])
    mitre_df = mitre_df.merge(mitre_df[['id', 'name']].rename(columns={'id':'par_id', 'name':'par_name'}), on='par_id', how='left')

    proc_df = mitre_df[['id', 'name', 'url', 'labels', 'proc_descr', 'kill_chain_tags', 'subtechnique', 'par_name']].explode('proc_descr')\
                      .rename(columns={'proc_descr':'sentence'})\
                      .assign(is_proc=True)
    
    main_descr_df = mitre_df[['id', 'name', 'url', 'labels', 'sentence', 'kill_chain_tags', 'subtechnique', 'par_name']].assign(is_proc=False)

    
    return mitre_df, main_descr_df, proc_df

def prep_mitr(main_descr_df, proc_df, conf):

    # предобработка
    # -------------------------------
    if conf['get_data']['divide_paras']:
        main_descr_df['sentence'] = main_descr_df['sentence'].str.split('\n')
        main_descr_df = main_descr_df.explode('sentence').loc[lambda x: x['sentence'].str.len()>0]
        
    mitre_attack_df = pd.concat([main_descr_df, proc_df], ignore_index=True)
    
    # удалим ссылочки
    mitre_attack_df['sentence'] = mitre_attack_df['sentence'].str.replace(r'\(https://[^\s]{1,}\)', r'', regex=True)
    mitre_attack_df['sentence'] = mitre_attack_df['sentence'].str.replace(r'\(Citation:[^\)]*\)', r'', regex=True)
    
    # удаляем строки, где процедуры не было совсем или citatation только (1 запись вроде)
    mitre_attack_df = mitre_attack_df[(~mitre_attack_df['sentence'].isna()) & (mitre_attack_df['sentence']!='')].reset_index(drop=True)

    return mitre_attack_df

