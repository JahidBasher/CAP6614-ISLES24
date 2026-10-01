#!/usr/bin/env bash
# Re-download the open-access sources used by the literature review and extract their text.
# data/papers/ and data/text/ are not committed (copyrighted full texts); the six abstract-only sources
# in data/text/ are committed because they cannot be re-fetched reliably.
# Usage: literature_review/scripts/fetch_sources.sh  (from any directory)     (needs curl and python3 with pypdf)
set -euo pipefail
cd "$(dirname "$0")/.."   # literature_review/
mkdir -p data/papers data/text
P=data/papers
UA="Mozilla/5.0 (research script)"

# arXiv preprints
while IFS=$'\t' read -r key id; do
  [ -s "$P/$key.pdf" ] && continue
  echo "arXiv $id -> $P/$key.pdf"
  curl -sfL -A "$UA" "https://arxiv.org/pdf/$id" -o "$P/$key.pdf"
  sleep 3   # be polite to arXiv
done < data/cache/manifest_arxiv.tsv

# Publisher open-access PDFs
[ -s $P/winzeck2018isles.pdf ] || curl -sfL -A "$UA" "https://www.frontiersin.org/articles/10.3389/fneur.2018.00679/pdf" -o $P/winzeck2018isles.pdf
[ -s $P/delarosa2025deepisles.pdf ] || curl -sfL -A "$UA" "https://www.nature.com/articles/s41467-025-62373-x.pdf" -o $P/delarosa2025deepisles.pdf

# PubMed Central full text (XML)
[ -s $P/hakim2021isles18.xml ] || curl -sf "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=8240494&rettype=xml" -o $P/hakim2021isles18.xml
[ -s $P/cegarra2026isles.xml ] || curl -sf "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=13517484&rettype=xml" -o $P/cegarra2026isles.xml

python3 scripts/extract.py 2>/dev/null
echo "Done. Now run: python3 scripts/verify.py"
