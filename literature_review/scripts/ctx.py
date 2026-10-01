import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run from literature_review/
import sys,re
from norm import pages
key,pat=sys.argv[1],sys.argv[2]; lim=int(sys.argv[3]) if len(sys.argv)>3 else 12
raw=open(f'data/text/{key}.txt').read(); n=0
for pg,t in pages(raw):
    sents=re.split(r'(?<=[.!?])\s+(?=[A-Z(\[])',t)
    for s in sents:
        if re.search(pat,s,re.I):
            print(f'[p{pg}] {s}\n'); n+=1
            if n>=lim: sys.exit()
