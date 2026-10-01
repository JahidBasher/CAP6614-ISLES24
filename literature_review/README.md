# Literature review

Evidence-linked review of final infarct prediction on ISLES'24 (41 papers, 128 verbatim excerpts).

```
report/     generated outputs, do not edit by hand
  literature_review.html   interactive: citations open the verbatim quote
  literature_review.md     same content as Markdown
content/    what you edit
  review_body.html         the review text; citations are [[paperkey#n]] markers
  template.html            page design for the HTML report
scripts/
  fetch_sources.sh         re-download open-access papers into data/papers and extract data/text
  extract.py               PDF / PMC XML -> data/text/<key>.txt
  build_corpus.py          paper list, verbatim excerpts, screening log -> data/excerpts.json
  verify.py                string-match every excerpt against its source text
  build_html.py            content + data -> report/literature_review.html
  build_md.py              content + data -> report/literature_review.md
  ctx.py                   search a paper's text for candidate quotes
  link_evidence.py         turn "(Ren 2025 #3)" or [[key#n]] citations in any .md into checked evidence links
  add_references.py        append/refresh a "## References" section from a .md file's evidence links
  norm.py                  text normalisation shared by the scripts
  biblio.py                venue and author formatting shared by the scripts
data/
  excerpts.json            metadata, excerpts, verification status, screening log
  cache/                   Semantic Scholar / Crossref / OpenAlex responses and the arXiv manifest
  text/                    extracted source text (only 6 abstract files are committed)
  papers/                  downloaded PDFs (not committed: copyrighted)
```

## Rebuild

Scripts can be run from any directory. Requires Python 3 with `pypdf`.

```bash
literature_review/scripts/fetch_sources.sh        # only needed once after cloning
python3 literature_review/scripts/build_corpus.py
python3 literature_review/scripts/verify.py       # must report FAIL: 0
python3 literature_review/scripts/build_html.py
python3 literature_review/scripts/build_md.py
```

## Adding a claim

1. Find a quote: `python3 literature_review/scripts/ctx.py <paperkey> "<regex>"`.
2. Add it with `ex(...)` under that paper in `scripts/build_corpus.py`. New excerpts get the next number, e.g. `ren2025isles24#8`.
3. Cite it in `content/review_body.html` as `[[ren2025isles24#8]]`.
4. Rebuild. `verify.py` rejects any quote that is not in the source text, and the build scripts reject unknown excerpt IDs.

## Citing evidence in other documents

Write citations as `(Ren 2025 #3)` or `[[ren2025isles24#3]]` in any Markdown file, then:

```bash
python3 literature_review/scripts/link_evidence.py methodology_proposal/your_file.md    # checks ids and numbers, inserts links
python3 literature_review/scripts/add_references.py methodology_proposal/your_file.md   # adds the References section
```
