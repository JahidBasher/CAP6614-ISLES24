"""Turn evidence citations in Markdown files into links to the verbatim quote in
report/literature_review.md, after checking that every id exists and that every number in a cited
sentence appears in one of its cited quotes. Already-converted links are left alone.
Two citation forms are accepted: [[paperkey#n]] markers, and plain author-year citations such as
"(Ren 2025 #3; de la Rosa 2024 #14)", where the label must match exactly one paper.

Usage: python3 literature_review/scripts/link_evidence.py methodology_proposal/some_file.md [more.md ...]
Exit code 1 if an unknown id is found; number mismatches are printed for manual review."""
import json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
D = json.load(open(os.path.join(ROOT, 'data', 'excerpts.json')))
EX = {e['id']: (p, e) for p in D['papers'] for e in p['excerpts']}
REPORT = os.path.join(ROOT, 'report', 'literature_review.md')
SPECIAL = {'delarosa': 'de la Rosa', 'devries': 'de Vries', 'cegarra': 'Martínez Cegarra', 'ajnr': 'Yang'}

def label(key):
    p = EX[next(i for i in EX if i.startswith(key + '#'))][0]
    name = next((v for k, v in SPECIAL.items() if key.startswith(k)), None)
    if not name:
        name = re.match(r'[a-z]+', key).group(0).capitalize()
    return f"{name} {p['year']}"

KEYS = sorted({i.split('#')[0] for i in EX})

def label_map():
    m = {}
    for k in KEYS:
        m.setdefault(label(k), []).append(k)
    dup = {l: ks for l, ks in m.items() if len(ks) > 1}
    if dup: sys.exit(f'ambiguous labels: {dup}')
    return {l: ks[0] for l, ks in m.items()}

PLAIN = re.compile(r'\(((?:[^()#;\[\]]+? \d{4} #\d+)(?:; [^()#;\[\]]+? \d{4} #\d+)*)\)')

def plain_to_markers(s, path):
    lm = label_map(); bad = []
    def conv(m):
        ids = []
        for part in m.group(1).split('; '):
            lab, n = part.rsplit(' #', 1)
            if lab not in lm: bad.append(lab); return m.group(0)
            ids.append(f'{lm[lab]}#{n}')
        return '[[' + '|'.join(ids) + ']]'
    s = PLAIN.sub(conv, s)
    if bad: print(f'{path}: unknown citation labels {sorted(set(bad))}')
    return s, not bad

def process(path):
    s, ok = plain_to_markers(open(path).read(), path)
    if not ok: return False
    ids = re.findall(r'\[\[([^\]]+)\]\]', s)
    unknown = sorted({i for m in ids for i in m.split('|') if i not in EX})
    if unknown:
        print(f'{path}: unknown excerpt ids {unknown}'); return False
    for sent in re.split(r'(?<=[.!?])\s+|\n', s):
        cited = [i for m in re.findall(r'\[\[([^\]]+)\]\]', sent) for i in m.split('|')]
        if not cited: continue
        txt = re.sub(r'\[\[[^\]]+\]\]', '', sent)
        quotes = ' '.join(EX[i][1]['quote'] for i in cited)
        nums = [n for n in re.findall(r'(?<![A-Za-z’\'])\d+(?:[.,]\d+)?', txt) if len(n) > 1 or '.' in n]
        miss = [n for n in nums if n not in quotes and n not in ('24', '2018', '2022', '2024')]
        if miss: print(f'{path}: check numbers {miss} in: {txt.strip()[:120]}')
    rel = os.path.relpath(REPORT, os.path.dirname(os.path.abspath(path)))
    def link(m):
        out = []
        for i in m.group(1).split('|'):
            k, n = i.split('#')
            out.append(f"[{label(k)} #{n}]({rel}#ex-{k}-{n})")
        return '; '.join(out)
    s = re.sub(r'\[\[([^\]]+)\]\]', lambda m: '(' + link(m) + ')', s)
    open(path, 'w').write(s)
    print(f'{path}: {len(ids)} evidence markers linked')
    return True

if __name__ == '__main__':
    ok = all([process(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
