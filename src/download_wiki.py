import wikipediaapi
import os

'''
Per eseguire lo script bisogna dichiarare user agent:
export WIKI_USER_AGENT='<NomeProgetto> (<email personale>)'; python download_wiki.py

Se si vogliono scaricare TANTE pagine wiki, EVITARE di usare questo metodo onde
non sovraccaricare l'API di Wikipedia, e usare invece i dump: https://it.wikipedia.org/wiki/Aiuto:Download_di_Wikipedia
'''

wiki = wikipediaapi.Wikipedia(
    user_agent=os.environ['WIKI_USER_AGENT'], 
    language='en',
)

page_name='Fishing_cat'
page = wiki.page(page_name)
print(page.text)
