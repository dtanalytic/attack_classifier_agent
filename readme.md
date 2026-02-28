# Настройка
- добавить данные. Создать папки:
    - data
    - data/external и положить:
        - файл enterprise-attack.json (json с mitre описаниями) 
        - файл multi_label.json (данные с репозиториями tram https://github.com/center-for-threat-informed-defense/tram/)
        - папку cti_hal (данные с репозитория https://github.com/dessertlab/CTI-HAL/tree/main/data)
        - отчеты компании в папку rep_dn       
- git clone https://github.com/dtanalytic/attack_classifier_agent
- настроить db (ноутбук notebooks/start.ipynb):
```
from src.prod.train import create_dbs
# result - files: faiss_db (data/prod/attack_db), bm25_retriever dump (data/prod/bm25_retriever.pkl), binarizer - (data/prod/mlb_ttp)
create_dbs()
```
# Инференс
- вызвать predict (ноутбук notebooks/start.ipynb):
```
from src.prod.predict import predict

pred_l = [''' Tor hidden service on a compromised system.''',
         '''DarkVishnya physically connected Bash Bunny, Raspberry Pi, netbooks, and inexpensive laptops to the target organization's environment to access the company’s local network''',
         '''Adversaries may abuse cloud management services to execute commands within virtual machines. Resources such as AWS Systems Manager, Azure RunCommand, and Runbooks allow users to remotely run scripts in virtual machines by leveraging installed virtual machine agents.''']
pred_df = predict(pred_l)

```