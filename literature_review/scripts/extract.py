import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run from literature_review/
import sys,os,re,html,json
from pypdf import PdfReader
os.makedirs('data/text',exist_ok=True)
out={}
for f in sorted(os.listdir('data/papers')):
    k,ext=os.path.splitext(f)
    p='data/papers/'+f
    if ext=='.pdf':
        try:
            r=PdfReader(p); pages=[]
            for i,pg in enumerate(r.pages):
                try: pages.append(pg.extract_text() or '')
                except Exception as e: pages.append('')
            txt=''.join(f'\n<<<PAGE {i+1}>>>\n'+t for i,t in enumerate(pages))
        except Exception as e:
            print('FAIL',f,e); continue
    elif ext=='.xml' and k in ('hakim2021isles18','cegarra2026isles'):
        d=open(p,encoding='utf-8').read()
        d=re.sub(r'<ref-list.*?</ref-list>','',d,flags=re.S)
        d=re.sub(r'</(p|title|abstract|sec)>','\n',d)
        txt=html.unescape(re.sub(r'<[^>]+>','',d))
    else: continue
    open(f'data/text/{k}.txt','w').write(txt)
    out[k]=len(txt.split())
for k,v in out.items(): print(f'{k:30s}{v:7d} words')
