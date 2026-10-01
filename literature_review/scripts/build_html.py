"""Assemble literature_review.html from review_body.html + excerpts.json.
Markers [[key#n|key#m]] become citation links; fails loudly on unknown excerpt ids."""
import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run from literature_review/
import json, re, html

D = json.load(open('data/excerpts.json'))
papers = {p['key']: p for p in D['papers']}
exmap = {e['id']: (p, e) for p in D['papers'] for e in p['excerpts']}
body = open('content/review_body.html').read()

from biblio import VENUE, venue, authors

# ---- numbering by first appearance
order = []
for m in re.finditer(r'\[\[([^\]]+)\]\]', body):
    for i in m.group(1).split('|'):
        if i not in exmap: raise SystemExit(f'Unknown excerpt id in body: {i}')
        k = exmap[i][0]['key']
        if k not in order: order.append(k)
uncited = [k for k in papers if k not in order]
for k in uncited: order.append(k)
num = {k: n + 1 for n, k in enumerate(order)}

def anchor(eid): return 'ex-' + eid.replace('#', '-')

def cite(m):
    ids = m.group(1).split('|')
    keys = []
    for i in ids:
        k = exmap[i][0]['key']
        if k not in keys: keys.append(k)
    label = ', '.join(str(num[k]) for k in keys)
    return (f'<a class="cite" href="#{anchor(ids[0])}" data-ids="{"|".join(ids)}" '
            f'aria-label="Show source quote for reference {label}">[{label}]</a>')
body_html = re.sub(r'\[\[([^\]]+)\]\]', cite, body)

# ---- word count of main body (sections 1-9, excluding the comparison table)
txt = re.sub(r'<table.*?</table>', ' ', body, flags=re.S)
txt = re.sub(r'\[\[[^\]]+\]\]', ' ', txt)
txt = html.unescape(re.sub(r'<[^>]+>', ' ', txt))
words = len(re.findall(r"[A-Za-z0-9'’.%±–-]+", txt))

def esc(s): return html.escape(str(s), quote=True)


def citation(p):
    parts = [f"{esc(authors(p))} ({p['year']}). {esc(p['title'])}. <em>{esc(venue(p))}</em>"]
    vi = ''
    if p.get('volume'): vi += f", {esc(p['volume'])}"
    if p.get('issue'): vi += f"({esc(p['issue'])})"
    if p.get('pages'): vi += f", {esc(p['pages'])}"
    parts.append(vi + '.')
    ids = []
    if p.get('doi') and not p['doi'].lower().startswith('10.48550'): ids.append(f'doi:{esc(p["doi"])}')
    if p.get('arxiv'): ids.append(f'arXiv:{esc(p["arxiv"])}')
    return ''.join(parts) + (' ' + ' · '.join(ids) if ids else '')

SECT = {'A': 'Benchmarks', 'B': "ISLES'24 methods & recent work", 'C': 'CTP outcome prediction',
        'D': 'Segmentation backbones', 'E': 'Video models', 'F': 'Foundation models & VLMs'}

# ---- paper list
rows = []
for k in order:
    p = papers[k]
    rows.append(f'''<tr id="ref-{num[k]}"><td class="num">{num[k]}</td>
<td class="refcell"><span class="reftitle">{esc(p['title'])}</span><br><span class="refmeta">{citation(p)}</span><br><a href="{esc(p['url'])}" target="_blank" rel="noopener">{esc(p['url'])}</a></td>
<td>{esc(venue(p))}</td><td class="num">{p['year']}</td><td class="num">{p['citations_s2']:,}</td>
<td><span class="chip sec-{p['section']}">{SECT[p['section']]}</span><br><span class="access">{esc(p['access'])}</span></td></tr>''')
paper_table = '\n'.join(rows)

# ---- evidence index
ev = []
for k in order:
    p = papers[k]
    items = ''.join(
        f'''<li id="{anchor(e['id'])}"><blockquote>{esc(e['quote'])}</blockquote>
<div class="exmeta"><code>{esc(e['id'])}</code> · {esc(e['loc'])} · <span class="vchip v-{e['verified']}">{'verbatim, exact match' if e['verified']=='exact' else 'verbatim, PDF spacing normalised'}</span></div></li>'''
        for e in p['excerpts'])
    if not items:
        items = '<li class="noquote">No accessible text. Listed for completeness and not cited for any claim.</li>'
    ev.append(f'''<div class="evpaper"><h4><span class="refno">[{num[k]}]</span> {esc(p['title'])}</h4>
<p class="refmeta">{esc(authors(p))} · {esc(venue(p))} · {p['year']} · <a href="{esc(p['url'])}" target="_blank" rel="noopener">source</a> · {esc(p['access'])}</p><ol class="exlist">{items}</ol></div>''')
evidence = '\n'.join(ev)

screen = '\n'.join(f'<tr><td>{esc(s["candidate"])}</td><td>{esc(s["found_via"])}</td><td>{esc(s["reason"])}</td></tr>' for s in D['screening_excluded'])
vc = D['verification']['counts']; total = sum(vc.values())
nq = sum(1 for p in D['papers'] if p['excerpts'])

# ---- data for the quote panel
panel = {e['id']: dict(q=e['quote'], loc=e['loc'], v=e['verified'], n=num[p['key']], t=p['title'],
                       a=authors(p, 1), y=p['year'], u=p['url'], acc=p['access'], anc=anchor(e['id']))
         for e_id, (p, e) in exmap.items()}

page = open('content/template.html').read()
for k, v in {'{{BODY}}': body_html, '{{PAPERS}}': paper_table, '{{EVIDENCE}}': evidence, '{{SCREEN}}': screen,
             '{{NPAPERS}}': str(len(papers)), '{{NQUOTED}}': str(nq), '{{NEX}}': str(total),
             '{{NEXACT}}': str(vc['exact']), '{{NWS}}': str(vc['whitespace']), '{{NFAIL}}': str(vc['FAIL']),
             '{{NSCREEN}}': str(len(papers) + 10), '{{WORDS}}': f'{words:,}',
             '{{PANELDATA}}': json.dumps(panel, ensure_ascii=False).replace('</', '<\\/')}.items():
    page = page.replace(k, v)
open('report/literature_review.html', 'w').write(page)
print(f'body words: {words}; papers: {len(papers)} (uncited in body: {uncited}); excerpts: {total}; '
      f'excerpts cited in body: {len(set(re.findall(r"[a-z0-9]+#[0-9]+", body)))}')
