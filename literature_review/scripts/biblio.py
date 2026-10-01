"""Bibliographic helpers shared by the build scripts: display venue and author lists for a paper record
from data/excerpts.json."""

_SWITCH = ('Image Analysis in Stroke Diagnosis and Interventions (SWITCH/ISLES 2024, MICCAI workshop), '
           'Lecture Notes in Computer Science')

# Venue names that Crossref / Semantic Scholar report too generically (e.g. only "Lecture Notes in Computer Science").
VENUE = {
    'mojtahedi2025switch': _SWITCH, 'amador2025switch': _SWITCH,
    'dalmazo2025switch': _SWITCH, 'urunuela2025switch': _SWITCH,
    'hatamizadeh2022swinunetr': 'BrainLes 2021 (MICCAI workshop), Lecture Notes in Computer Science',
    'wang2023sammed3d': 'ECCV Workshops, Lecture Notes in Computer Science',
    'perera2024segformer3d': 'CVPR Workshops (CVPRW) 2024',
    'liu2022videoswin': 'CVPR 2022', 'liu2023clipdriven': 'ICCV 2023',
}


def venue(p):
    return VENUE.get(p['key']) or p['venue'] or ''


def authors(p, n=3):
    a = p['authors']
    return ', '.join(a[:n]) + (', et al.' if len(a) > n else '')


def identifiers(p):
    ids = []
    if p.get('doi') and not p['doi'].lower().startswith('10.48550'):
        ids.append(f"doi:{p['doi']}")
    if p.get('arxiv'):
        ids.append(f"arXiv:{p['arxiv']}")
    return ids


def volume_pages(p):
    return ((f", {p['volume']}" if p.get('volume') else '') + (f"({p['issue']})" if p.get('issue') else '')
            + (f", {p['pages']}" if p.get('pages') else ''))
