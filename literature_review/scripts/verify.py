"""Verify every excerpt in excerpts.json occurs in its source text.
Levels: 'exact' = substring of normalized text (NFKC, ligatures, de-hyphenated line breaks, collapsed whitespace);
'whitespace' = identical after removing ALL whitespace (fixes PDF extraction spacing like 'o utcome');
'FAIL' = not found."""
import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run from literature_review/
import json,re,sys
from norm import norm
D=json.load(open('data/excerpts.json'))
cache={}
def src(f):
    if f not in cache:
        raw=open(f).read()
        a=norm(raw); b=norm(raw,dehyph=False)
        cache[f]=(a,b,re.sub(r'\s+','',a),re.sub(r'\s+','',b))
    return cache[f]
stats={'exact':0,'whitespace':0,'FAIL':0}; fails=[]
for p in D['papers']:
    for e in p['excerpts']:
        q=norm(e['quote']); a,b,an,bn=src(p['text_file'])
        if q in a or q in b: lvl='exact'
        elif re.sub(r'\s+','',q) in an or re.sub(r'\s+','',q) in bn: lvl='whitespace'
        else: lvl='FAIL'; fails.append((e['id'],e['quote'][:90]))
        e['verified']=lvl; stats[lvl]+=1
D['verification']=dict(method=__doc__.strip(), counts=stats, date='2026-09-30')
json.dump(D,open('data/excerpts.json','w'),indent=1,ensure_ascii=False)
print(stats)
for f in fails: print('FAIL',f)
