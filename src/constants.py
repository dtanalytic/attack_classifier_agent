import re
import logging
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

from typing import List

LOG_FN = 'journal.log'


system_template = (
        "You are an expert in cybersecurity analysis."
        "Analyze a given text for the MITRE ATT&CK tactics and techniques described or mentioned in it."
        "Carefully examine the applicability based on the context - "
        "Linux or Windows, what language is used, what software impacted. Only include relevant TTPs."
        "You will be provided with rag data after key phrase 'Known information:', keep it in mind to better classify input text"
        "Each example in rag will be provided with metadata: Techniques, Tactics, MITRE title, MITRE url. If text contains no malicios content, metadata will be filled with nans"
        "Return MITRE ATT&CK IDs, corresponding TTP names, and reasoning according to instructions:{parser_instructions}"
        "Example:\n"
        '[{{"mitre_id": "T1001", "mitre_name":"Data Obfuscation", "reason":"an exact extract from the analyzed text'
        'that acts as a proof and demonstrates the usage of the TTP"}}]'
        'If no TTPs are found, return an empty array like this:\n[]'
    )

user_template = '''
Known information:
{context}
Based on the above known information (rag), respond to the user's question with a json described in system prompt.
Text to analyse:
{input}
'''

user_bert_template = '''Custom model predictions:
{model_preds}
Known information:
{context}
Based on the above known information (rag), and custom model predictions (may be wrong) respond to the user's question with a json described in system prompt.
Text to analyse:
{input}
'''

document_prompt = PromptTemplate.from_template('Contents:"{page_content}", Techniques - {techniques}, Tactics - {tactics}, MITRE title - {name}, MITRE url - {url}')


class TTP(BaseModel):
    mitre_id: str = Field(..., description='MITRE ATT&CK TTP code', pattern=r'^(T\d{4}(?:\.\d{3})?)$')
    mitre_name: str = Field(..., description='Title of MITRE ATT&CK TTP code', min_length=2, strict=True)
    reason: str = Field(..., description='Exact part of the input text where MITRE ATT&CK TTP was found', min_length=2, strict=True)

class TTPList(BaseModel):
    ttps: List[TTP]


regexp_fname = re.compile(r'[a-zA-Z0-9_-]+\.[a-zA-Z0-9_]{3,}')
regexp_fname_upd = re.compile(r'\b[.a-zA-Z0-9_-]+\.(?=[a-zA-Z0-9_]*[a-zA-Z])[a-zA-Z0-9_]{2,}\b')

regexp_fpath = re.compile(r'(([a-zA-Z]:[\\\/]+)|([\/]))?(?:[a-zA-Z0-9_-]+[\\\/])*[a-zA-Z0-9_-]+\.[a-zA-Z0-9_]{3,}')
regexp_fpath_upd = re.compile(
    r'''
(^|\s|,)(
   (?:[\w]:[\\/]+|(?:[\\/]{1,2})|\./) 
   (?:[\w._-]+[\\/]+){1,}            
   [\w._-]+                        
   (?:\.[\w]+)?                    
)\b
''',
    re.VERBOSE
)


regexp_cve = re.compile(r'CVE-\d{4}-\d{4,}')
regexp_cve_upd = re.compile(r'\bCVE-(19|20)\d{2}-\d{4,}\b', re.IGNORECASE)

regexp_ipv4 = re.compile("^((((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|" +
                         "[01]?[0-9][0-9]?))(:\\d+)?)$")

regexp_ipv4_upd = re.compile(r'''
                          \b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}
                          (25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
                          (:\d+)?\b
                         ''',
                            re.VERBOSE)

regexp_ipv6 = re.compile("^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|" +
                         "([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}" +
                         "(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|" +
                         "([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}" +
                         "(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|" +
                         ":((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|" +
                         "::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|" +
                         "(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|" +
                         "(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$")

regexp_ipv6_upd = re.compile(r"""
\b
(
    ([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4} |
    ([0-9a-fA-F]{1,4}:){1,7}: |
    ([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4} |
    ([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2} |
    ([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3} |
    ([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4} |
    ([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5} |
    [0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6}) |
    :((:[0-9a-fA-F]{1,4}){1,7}|:) |
    fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]+ |
    ::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1?[0-9])?[0-9])\.){3}(25[0-5]|(2[0-4]|1?[0-9])?[0-9]) |
    ([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1?[0-9])?[0-9])\.){3}(25[0-5]|(2[0-4]|1?[0-9])?[0-9])
)
\b
""", re.VERBOSE)
# regexp_domain = re.compile("^((((?=[a-z0-9\\-_]{1,63}\\.)(xn--)?[a-z0-9_\\-]+(-[a-z0-9_]+)*\\.)+" +
#                            "[a-z-0-9]{2,63})(:\\d+)?)$")

regexp_domain = re.compile(r'([a-zA-Z0-9-]+\.){2,}[a-zA-Z]{2,}')
regexp_domain_upd = re.compile(r"""
    \b                        
    ([\w-]+\.){1,}      
    [\w]{2,}            
    \b                 
""", re.VERBOSE)

regex_domain_zone = re.compile("\\.(([a-z]+)|(xn--[a-z0-9]+))(:\d+)?$")
regex_domain_zone_upd = re.compile(r'''
    \b
    (?:xn--[\w]+)
    (?:\.[\w-]{2,})+
    (?::\d+)?
    \b
''', re.VERBOSE | re.IGNORECASE)


regexp_email = re.compile("^((?:[a-z0-9!#\$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#\$%&'*+/=?^_`{|}~-]+)*|" +
                          "\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\" +
                          "[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(((?=[a-z0-9\\-_]{1,63}\\.)" +
                          "(xn--)?[a-z0-9_\\-]+(-[a-z0-9_]+)*\\.)+[a-z-0-9]{2,63}))$")

regexp_email_upd = re.compile(r"""
    (                           
        (?:                     
            [a-z0-9!#$%&'*+/=?^_`{|}~-]+  
            (?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*  
            |                   
            "(?:                
                [\x01-\x08\x0b\x0c\x0e-\x21\x23-\x5b\x5d-\x7f] 
                |\\[\x01-\x09\x0b\x0c\x0e-\x7f]                
            )*"
        )
    )
    @                           
    (                           
        (?=                    
            [a-z0-9-]{1,63}\.
        )
        (?:                    
            (xn--)?            
            [a-z0-9-]+         
            (?:\-[a-z0-9_]+)*  
            \.                 
        )+
        [a-z0-9]{2,63}         
    )
""", re.VERBOSE | re.IGNORECASE)


regexp_hash_md5 = re.compile(r"^([a-f0-9]{32})$")
regexp_hash_md5_upd = re.compile(r"([a-f0-9]{32})", re.IGNORECASE)

regexp_hash_sha1 = re.compile(r"^([a-f0-9]{40})$")
regexp_hash_sha1_upd = re.compile(r"([a-f0-9]{40})", re.IGNORECASE)

regexp_hash_sha256 = re.compile(r"^([a-f0-9]{64})$")
regexp_hash_sha256_upd = re.compile(r"([a-f0-9]{64})", re.IGNORECASE)

regexp_hash_sha512 = re.compile(r"^([a-f0-9]{128})$")
regexp_hash_sha512_upd = re.compile(r"([a-f0-9]{128})", re.IGNORECASE)

regexp_hash_ssdeep = re.compile(r"^(\d+(:[+/a-zA-Z0-9]{5,})+)$")
regexp_hash_ssdeep_upd = re.compile(r"(\d+(:[+/a-zA-Z0-9]{5,})+)", re.IGNORECASE)



regexp_url = re.compile("^((?:(?:(?:https?|ftps?|tcp|udp):)?\\/\\/)(?:\\S+(?::\\S*)?@)?(?:(?!(?:10|127)" +
                        "(?:\\.\\d{1,3}){3})(?!(?:169\\.254|192\\.168)(?:\\.\\d{1,3}){2})(?!172\\." +
                        "(?:1[6-9]|2\\d|3[0-1])(?:\\.\\d{1,3}){2})(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])" +
                        "(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[1-9]\\d?|1\\d\\d|2[0-4]\\d|" +
                        "25[0-4]))|(?:(?:[a-z0-9\\\\u00a1-\\\\uffff][a-z0-9\\\\u00a1-\\\\uffff_-]" +
                        "{0,62})?[a-z0-9\\\\u00a1-\\\\uffff]\\.)+(?:[a-z\\\\u00a1-\\\\uffff]{2,}|" +
                        "xn--[a-z0-9]+\\.?))(?::\\d{2,5})?(?:[/?#]\\S*)?)$")

regexp_url_upd = re.compile(
    r"""
    (?:
        (?:
            (?: https? | ftps? | tcp | udp ) :  
        )?
        //                                      
    )
    (?: \S+ (?: : \S* )? @ )?                   

    (?:
        
        (?! (?:10|127) (?: \. \d{1,3} ){3} )
        (?! (?:169\.254|192\.168) (?: \. \d{1,3} ){2} )
        (?! 172 \. (?:1[6-9]|2\d|3[0-1]) (?: \. \d{1,3} ){2} )
        (?: [1-9]\d? | 1\d\d | 2[01]\d | 22[0-3] )
        (?: \. (?: 1?\d{1,2} | 2[0-4]\d | 25[0-5] ) ){2}
        (?: \. (?: [1-9]\d? | 1\d\d | 2[0-4]\d | 25[0-4] ) )

        |
        
        (?:
            (?: [a-z0-9\u00a1-\uffff] [a-z0-9\u00a1-\uffff_-]{0,62} )?
            [a-z0-9\u00a1-\uffff] \.
        )+
        (?: [a-z\u00a1-\uffff]{2,} | xn--[a-z0-9]+ \.? )
    )

    (?:: \d{2,5} )?                             
    (?: [/?#] \S* )?
    """,
    re.IGNORECASE | re.VERBOSE
)



regexp_registry = re.compile('^((hkey_local_machine|hkey_classes_root|hkey_current_user|hkey_users|hkey_current_config|'
                             'hklm|hlm|hkcr|hcr|hkcu|hcu|hkcc|hcc)[ ]?[/\\][ ]?'
                             '([a-z0-9\s_@\-\^!#.\:\/\$%&+={}\[\]\\* ])+)$')


# new version
regexp_registry_upd = re.compile(r'''\b((hkey_local_machine|hkey_classes_root|hkey_current_user|hkey_users|hkey_current_config|
                         hklm|hlm|hkcr|hcr|hkcu|hcu|hkcc|hcc)
                         (\s?[\/\\]\s?[\w\s_@\-\^!#.\:\/\$%&+={}\[\]\\* ]+)?)\b''', 
                 flags=re.IGNORECASE | re.VERBOSE)

regexp_coins_eth = re.compile(r'^(0x[a-f0-9]{40})$')
regexp_coins_btc = re.compile(r'^((bc1[a-za-hj-np-z0-9]{35,99})|([3]{1}[a-km-za-hj-np-z1-9]{25,34}))$')
regexp_coins_bch = re.compile(r'^(((bitcoincash|bchreg|bchtest):)?[qp][a-z0-9]{41})$')
regexp_coins_ltc = re.compile(r'^(([lm]{1}[a-km-za-hj-np-z1-9]{25,34})|(ltc1[a-za-hj-np-z0-9]{35,99}))$')
regexp_coins_doge = re.compile(r'^(d[5-9a-hj-np-u]{1}[1-9a-hj-np-za-km-z]{32})$')
regexp_coins_dash = re.compile(r'^(x[1-9a-hj-np-za-km-z]{33})$')
regexp_coins_xmr = re.compile(r'^([84][0-9a-b]{1}[1-9a-hj-np-za-km-z]{93,117})$')
regexp_coins_neo = re.compile(r'^(a[0-9a-z]{33})$')
regexp_coins_xrp = re.compile(r'^(r[0-9a-z]{33})$')

regexp_coins_eth_upd = re.compile(r'(0x[a-f0-9]{40})')
regexp_coins_btc_upd = re.compile(r'((bc1[a-za-hj-np-z0-9]{35,99})|([3]{1}[a-km-za-hj-np-z1-9]{25,34}))')
regexp_coins_bch_upd = re.compile(r'(((bitcoincash|bchreg|bchtest):)?[qp][a-z0-9]{41})')
regexp_coins_ltc_upd = re.compile(r'(([lm]{1}[a-km-za-hj-np-z1-9]{25,34})|(ltc1[a-za-hj-np-z0-9]{35,99}))')
regexp_coins_doge_upd = re.compile(r'(d[5-9a-hj-np-u]{1}[1-9a-hj-np-za-km-z]{32})')
regexp_coins_dash_upd = re.compile(r'(x[1-9a-hj-np-za-km-z]{33})')
regexp_coins_xmr_upd = re.compile(r'([84][0-9a-b]{1}[1-9a-hj-np-za-km-z]{93,117})')

regexp_coins_neo_upd = re.compile(r'\b(a(?=[0-9a-z]*[0-9])[0-9a-z]{33})\b')
regexp_coins_xrp_upd = re.compile(r'\b(r(?=[0-9a-z]*[0-9])[0-9a-z]{33})\b')



class ActivityLogger():

    def __init__(self, level=logging.INFO, encoding='utf-8'):

        self.level = level
        self.file_handler = logging.FileHandler(LOG_FN, encoding=encoding)
        self.console_handler = logging.StreamHandler()

    def get_logger(self, name):

        file_formatter = logging.Formatter(f'%(asctime)s \nMODULE:{name}, LEVEL:%(levelname)s, LINE:%(lineno)s, MSG:%(message)s\n')
        cons_formatter = logging.Formatter(f'%(asctime)s \n%(lineno)s %(message)s\n\n')

        # Применяем форматтер к обоим Handler
        self.file_handler.setFormatter(file_formatter)
        self.console_handler.setFormatter(cons_formatter)

        # Создаем объект logger и применяем к нему оба Handler
        logger = logging.getLogger(name)
        logger.setLevel(self.level)

        if len(logger.handlers)==0:
            logger.addHandler(self.file_handler)
            logger.addHandler(self.console_handler)

        # Disable propagation of messages to the root logger
        logger.propagate = False

        return logger

    def close_logger(self, logger):

        # import pdb;pdb.set_trace()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        self.file_handler.close()
        self.console_handler.close()