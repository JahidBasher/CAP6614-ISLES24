import re,unicodedata
LIG={'ﬀ':'ff','ﬁ':'fi','ﬂ':'fl','ﬃ':'ffi','ﬄ':'ffl','­':'','‐':'-','‑':'-','‒':'-','–':'–','−':'-',' ':' '}
def norm(s, dehyph=True):
    s=unicodedata.normalize('NFKC',s)
    for a,b in LIG.items(): s=s.replace(a,b)
    s=re.sub(r'<<<PAGE \d+>>>',' ',s)
    if dehyph: s=re.sub(r'(\w)-\s*\n\s*(\w)',r'\1\2',s)
    s=re.sub(r'\s+',' ',s)
    return s.strip()
def pages(raw):
    """yield (page, normalized text)"""
    parts=re.split(r'<<<PAGE (\d+)>>>',raw)
    if len(parts)==1: yield (None,norm(raw)); return
    for i in range(1,len(parts),2): yield (int(parts[i]),norm(parts[i+1]))
