"""Assemble literature_review.md from review_body.html + excerpts.json (same sources as the HTML).
Citations become [n] links to the excerpt anchors in the evidence appendix; the link title carries the
verbatim quote so it shows on hover in most Markdown viewers."""
import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run from literature_review/
import json, re, html

D = json.load(open('data/excerpts.json'))
papers = {p['key']: p for p in D['papers']}
exmap = {e['id']: (p, e) for p in D['papers'] for e in p['excerpts']}
body = open('content/review_body.html').read()

from biblio import VENUE, venue, authors
SECT = {'A': 'Benchmarks', 'B': "ISLES'24 methods & recent work", 'C': 'CTP outcome prediction',
        'D': 'Segmentation backbones', 'E': 'Video models', 'F': 'Foundation models & VLMs'}

order = []
for m in re.finditer(r'\[\[([^\]]+)\]\]', body):
    for i in m.group(1).split('|'):
        if i not in exmap: raise SystemExit(f'Unknown excerpt id: {i}')
        if exmap[i][0]['key'] not in order: order.append(exmap[i][0]['key'])
order += [k for k in papers if k not in order]
num = {k: n + 1 for n, k in enumerate(order)}
anchor = lambda eid: 'ex-' + eid.replace('#', '-')

def cite(m):
    out = []
    seen = set()
    for i in m.group(1).split('|'):
        p, e = exmap[i]
        q = e['quote'].replace('"', "'").replace('\n', ' ')
        lab = str(num[p['key']])
        # one link per excerpt; repeated paper numbers get a sub-index so each quote stays reachable
        lab = lab if p['key'] not in seen else lab + '′'
        seen.add(p['key'])
        out.append(f'[{lab}](#{anchor(i)} "{q}")')
    return '<sup>[' + ', '.join(out) + ']</sup>'

def inline(s):
    s = re.sub(r'<span class="ours-tag">(.*?)</span>\s*', r'**▸ \1:** ', s)
    s = re.sub(r'<strong>(.*?)</strong>', r'**\1**', s, flags=re.S)
    s = re.sub(r'<em>(.*?)</em>', r'*\1*', s, flags=re.S)
    s = re.sub(r'<a href="([^"]+)">(.*?)</a>', r'[\2](\1)', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'\s+', ' ', html.unescape(s)).strip()
    return re.sub(r'\[\[([^\]]+)\]\]', cite, s)

out = []
def emit(x=''): out.append(x)

# --- body conversion (structure of review_body.html is known and simple)
for sec in re.findall(r'<section id="([^"]+)">(.*?)</section>', body, re.S):
    sid, sb = sec
    for tok in re.finditer(r'<h2>(.*?)</h2>|<p( class="ours")?>(.*?)</p>|<(ul|ol)([^>]*)>(.*?)</\4>|<div class="table-wrap".*?<table[^>]*>(.*?)</table>\s*</div>', sb, re.S):
        if tok.group(1) is not None:
            h = re.sub(r'<span class="secno">(.*?)</span>', r'\1. ', tok.group(1))
            emit(f'<a id="{sid}"></a>\n\n## {inline(h)}\n')
        elif tok.group(3) is not None:
            t = inline(tok.group(3))
            emit(('> ' + t if tok.group(2) else t) + '\n')
        elif tok.group(4) is not None:
            ours = 'ours' in tok.group(5)
            ordered = tok.group(4) == 'ol'
            lis = re.findall(r'<li>(.*?)</li>', tok.group(6), re.S)
            for n, li in enumerate(lis, 1):
                bullet = f'{n}.' if ordered else '-'
                emit(('> ' if ours else '') + f'{bullet} {inline(li)}')
            emit()
        elif tok.group(7) is not None:
            t = tok.group(7)
            cap = inline(re.search(r'<caption>(.*?)</caption>', t, re.S).group(1))
            emit(f'*Table: {cap}*\n')
            heads = [inline(h) for h in re.findall(r'<th scope="col">(.*?)</th>', t)]
            emit('| ' + ' | '.join(heads) + ' |'); emit('|' + '---|' * len(heads))
            for row in re.findall(r'<tr[^>]*>(.*?)</tr>', re.search(r'<tbody>(.*?)</tbody>', t, re.S).group(1), re.S):
                g = re.search(r'<th colspan="\d+"[^>]*>(.*?)</th>', row)
                if g: emit(f'| **{inline(g.group(1))}** |' + ' |' * (len(heads) - 1)); continue
                cells = [inline(c).replace('|', '\\|') for c in re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)]
                emit('| ' + ' | '.join(cells) + ' |')
            emit()

body_md = '\n'.join(out)
words = len(re.findall(r"[A-Za-z0-9'’.%±–-]+", re.sub(r'<sup>.*?</sup>', ' ', re.sub(r'\n\|.*', '', body_md))))

# --- appendices
vc = D['verification']['counts']; total = sum(vc.values())
nq = sum(1 for p in D['papers'] if p['excerpts'])
A = []
A.append('<a id="papers"></a>\n\n## A. Paper list\n')
A.append('Numbered in order of first citation. Venue, volume and pages come from Crossref when the paper has a DOI, otherwise from Semantic Scholar or the arXiv listing. "Cited by" counts come from the Semantic Scholar API and were retrieved on 30 September 2026. Very recent papers will show low counts.\n')
A.append('| # | Short reference | Published venue | Year | Cited by | Topic |')
A.append('|---|---|---|---|---|---|')
for k in order:
    p = papers[k]
    A.append(f"| {num[k]} | {authors(p,1)} | {venue(p)} | {p['year']} | {p['citations_s2']:,} | {SECT[p['section']]} |")
A.append('\n### Full citations\n')
for k in order:
    p = papers[k]
    vi = (f", {p['volume']}" if p.get('volume') else '') + (f"({p['issue']})" if p.get('issue') else '') + (f", {p['pages']}" if p.get('pages') else '')
    ids = []
    if p.get('doi') and not p['doi'].lower().startswith('10.48550'): ids.append(f"doi:{p['doi']}")
    if p.get('arxiv'): ids.append(f"arXiv:{p['arxiv']}")
    A.append(f"<a id=\"ref-{num[k]}\"></a>**[{num[k]}]** {authors(p)} ({p['year']}). {p['title']}. *{venue(p)}*{vi}."
             + (f" {' · '.join(ids)}." if ids else '') + f" <{p['url']}>  \n"
             f"Cited by {p['citations_s2']:,} (Semantic Scholar). Access: {p['access']}. Inclusion: {p['inclusion_rule']}.\n")

A.append('<a id="evidence"></a>\n\n## B. Verbatim evidence\n')
A.append('Every excerpt cited above, grouped by paper. Excerpt IDs match `excerpts.json`. Page numbers refer to the downloaded PDF. For PubMed Central XML and abstract-only sources, the location is a section name.\n')
for k in order:
    p = papers[k]
    A.append(f"### [{num[k]}] {p['title']}\n")
    A.append(f"{authors(p)} · *{venue(p)}* · {p['year']} · <{p['url']}>  \nAccess: {p['access']}\n")
    if not p['excerpts']:
        A.append('*No accessible text. Listed for completeness and not cited for any claim.*\n')
    for e in p['excerpts']:
        v = 'verbatim, exact match' if e['verified'] == 'exact' else 'verbatim, PDF spacing normalised'
        A.append(f"<a id=\"{anchor(e['id'])}\"></a>\n> {e['quote']}\n>\n> — `{e['id']}` · {e['loc']} · {v}\n")

A.append('<a id="screening"></a>\n\n## C. Screening log: excluded candidates\n')
A.append("Inclusion rules: R1, reports on ISLES'24 or describes it; R2, predicts final infarct or core from acute CT/CTP; R3, original source of a method we rely on; R4, evidence for a specific claim.\n")
A.append('| Candidate | Found via | Reason for exclusion |'); A.append('|---|---|---|')
for s in D['screening_excluded']:
    A.append(f"| {s['candidate']} | {s['found_via']} | {s['reason']} |")

A.append('\n<a id="verification"></a>\n\n## D. Verification\n')
A.append("Each source was downloaded and its text extracted locally from the arXiv or publisher PDF, PubMed Central XML, or the PubMed abstract. Excerpts were copied from that text, not paraphrased by a model. A script then checked each excerpt against the source after Unicode normalisation, ligature expansion, re-joining of words hyphenated across line breaks, and whitespace collapsing.\n")
A.append(f"**{vc['exact']}** excerpts matched exactly. **{vc['whitespace']}** matched once all whitespace was ignored: the PDF extractor had split or merged words, for example \"lesion o utcome\". In those cases the characters are identical. **{vc['FAIL']}** failed.\n")
A.append("What verification does not cover: it proves that each quote exists in its source. It does not prove that the surrounding sentence in this review characterises the quote fairly, so judge that by reading the quote. Two sources are abstracts taken from non-publisher pages: Mojtahedi et al. from the Amsterdam UMC repository, and Amador et al. 2024 from the authors' GitHub README. The ISLES'24 dataset paper is quoted from its arXiv preprint (2408.11142), because the Radiology: AI version is paywalled.\n")

head = f"""# Predicting the Final Infarct on ISLES'24: Literature Review

*CAP6614 project · 30 September 2026*

What the literature says about predicting post-treatment infarcts from acute CT, whether perfusion time series and video encoders help, and where vision-language models fit.

**{len(papers)}** papers included (of {len(papers)+10} screened) · **{nq}** quoted · **{total}** verbatim excerpts · **{vc['FAIL']}** unverified · main body **{words:,}** words

**How to read this document.** A citation such as <sup>[3]</sup> links to the verbatim quote in [Appendix B](#evidence). In most Markdown viewers, hovering over the citation shows the quote. Paragraphs in block quotes that begin with **▸ Our reading:** are our interpretation, not a claim from any source.

**Contents:** [1 Scope](#scope) · [2 Lineage](#lineage) · [3 State of the art](#sota) · [4 Post-challenge work](#recent) · [5 Temporal perfusion](#temporal) · [6 Backbones](#backbones) · [7 Video models](#video) · [8 Foundation models & VLMs](#vlm) · [9 Gaps](#gaps) · [A Paper list](#papers) · [B Evidence](#evidence) · [C Screening](#screening) · [D Verification](#verification)

---
"""
open('report/literature_review.md', 'w').write(head + '\n' + body_md + '\n---\n\n' + '\n'.join(A) + '\n')
leftover = re.findall(r'<(?!a id=|/a>|sup>|/sup>)[a-z/][^>]*>', body_md)
print(f'words: {words}; leftover html tags in body: {leftover[:5]}')
