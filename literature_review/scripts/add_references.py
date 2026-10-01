"""Append (or refresh) a "## References" section at the end of Markdown files, listing every paper
cited through evidence links (literature_review.md#ex-<key>-<n>), formatted from data/excerpts.json.
Run link_evidence.py first. Safe to re-run: an existing References section at the end is replaced.

Usage: python3 literature_review/scripts/add_references.py methodology_proposal/some_file.md [more.md ...]"""
import os, re, sys
from link_evidence import EX, REPORT, label
from biblio import venue, authors, identifiers, volume_pages

PAPERS = {p['key']: p for p, _ in EX.values()}
HEADING = '## References'


def references(path, s):
    cited = {}
    for key, n in re.findall(r'literature_review\.md#ex-([a-z0-9]+)-(\d+)\)', s):
        cited.setdefault(key, set()).add(int(n))
    unknown = [k for k in cited if k not in PAPERS]
    if unknown:
        sys.exit(f'{path}: unknown paper keys {unknown}')
    rel = os.path.relpath(REPORT, os.path.dirname(os.path.abspath(path)))
    lines = [HEADING, '',
             'Papers cited in this document. Bibliographic details come from Crossref when a DOI exists, '
             'otherwise from Semantic Scholar or arXiv. "Quotes" link to the verbatim excerpts used here.', '']
    for key in sorted(cited, key=lambda k: label(k).lower()):
        p = PAPERS[key]
        ids = [i for i in identifiers(p) if i.split(':', 1)[1] not in venue(p)]
        title = p['title'] if p['title'][-1] in '.?!' else p['title'] + '.'
        quotes = ', '.join(f'[#{n}]({rel}#ex-{key}-{n})' for n in sorted(cited[key]))
        lines.append(f"- **{label(key)}.** {authors(p)} ({p['year']}). {title} *{venue(p)}*{volume_pages(p)}."
                     + (f" {' · '.join(ids)}." if ids else '') + f" <{p['url']}>  \n  Quotes: {quotes}")
    return '\n'.join(lines) + '\n', len(cited)


def process(path):
    s = open(path).read()
    i = s.rfind('\n' + HEADING + '\n')
    body = (s[:i] if i >= 0 else s).rstrip('\n')
    refs, n = references(path, body)
    open(path, 'w').write(body + '\n\n---\n\n' + refs if not body.endswith('---') else body + '\n\n' + refs)
    print(f'{path}: {n} references')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        process(p)
