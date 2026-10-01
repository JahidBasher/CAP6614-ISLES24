# CAP6614 Project Proposal: Final Infarct Prediction on ISLES'24

Group project for CAP6614 (Mr. Waves, Sara, Jahid, Raiyaan). We predict the post-treatment stroke infarct from pre-treatment CT using the ISLES'24 dataset. We test whether a pretrained video encoder over the perfusion maps improves on the challenge-winning nnU-Net baseline.

**Proposal due: 2 October 2026.** Requirements: 1–2 pages excluding references, PDF, about 3–5 references. See [the assignment](project_guideline/Project-proposal.pdf). Every team member submits the same PDF.

## Proposal status

| § | Section | Owner | File | Status |
|---|---|---|---|---|
| — | Title, authors, abstract | unassigned | `overleaf/main.tex`, `overleaf/sections/review.tex` | Title and authors need updating (only one author listed). Abstract empty. |
| 2 | Problem and motivation | Mr. Waves | `overleaf/sections/review.tex` | Empty. Source material: [initial_problem_statement.md](methodology_proposal/initial_problem_statement.md) (prepared by claude based on existing section)|
| 3 | Related work and research gap | Sara | `overleaf/sections/review.tex` | Drafted |
| 4 | Proposed method | Jahid | `overleaf/sections/method.tex` | Drafted (version 1, with TikZ figure) |
| 5 | Dataset and experiments | Jahid | `overleaf/sections/method.tex` | Drafted (version 1) |
| 6 | Novelty | Raiyaan | `overleaf/sections/review.tex` | Drafted |
| 7 | Feasibility and fallback plan | Raiyaan | `overleaf/sections/review.tex` | Drafted |
| 8 | Timeline and responsibilities | unassigned | `overleaf/sections/review.tex` | Missing. Must list each member's responsibilities. |

**Open issues before submission**
- **Page count and figure.** Check the page count (2-page limit) and the TikZ figure in the Overleaf build.
- **References.** `main.bib` has 10 entries, while the assignment asks for about 3–5. The team should agree what to keep.
- **Research gap vs. novelty.** The gap in §3 (value of extra inputs and clinical data) doesn't match the novelty in §6 (video encoders). See [00_selection.md](methodology_proposal/methodology_proposals/00_selection.md) for options.
- **Problem statement.** `initial_problem_statement.md` needs three corrections before use:
  - the treatment is described as "intracranial interventional reperfusion therapy" in the dataset paper
  - the sources disagree on the recanalisation grade: TICI 2b/3 in the challenge paper vs. mTICI 2c/3 in the dataset preprint
  - narrow its research question to match the chosen method

## Where to start

1. **[Literature review](literature_review/report/literature_review.md):** 41 papers. Every factual claim links to a verbatim quote from the source paper. GitHub renders this Markdown version. The [interactive HTML version](literature_review/report/literature_review.html) opens quotes in a side panel, but GitHub shows it as source code, so download it and open it in a browser.
2. **[Methodology guide](methodology_proposal/methodology_guide.md):** all 26 method options the literature supports, with effort, risk and suggested role.
3. **[Top-5 methodology proposals](methodology_proposal/methodology_proposals/00_selection.md):** about 1,500 words each, with references:
   - [P1 native 4D CTP as video](methodology_proposal/methodology_proposals/P1_native_ctp_video.md)
   - [P2 perfusion maps as video](methodology_proposal/methodology_proposals/P2_perfusion_maps_video.md) (the current §4–5)
   - [P3 text-conditioned clinical fusion](methodology_proposal/methodology_proposals/P3_text_conditioned_clinical_fusion.md)
   - [P4 MedSAM2 refinement](methodology_proposal/methodology_proposals/P4_medsam2_slice_video_refinement.md)
   - [P5 preprocessing and modality study](methodology_proposal/methodology_proposals/P5_preprocessing_modality_study.md)

## Repository layout

```
overleaf/                 LaTeX source of the proposal
  main.tex                entry point / Overleaf main document
  preamble.tex            extra packages and \TODO macros
  main.bib                bibliography (the only .bib that is compiled)
  iccv.sty, ieeenat_fullname.bst   ICCV 2025 author-kit style files
  sections/
    review.tex            proposal body; \input{sections/method} pulls in sections 4-5
    method.tex            Proposed method + Dataset and experiments (version 1), incl. TikZ figure
methodology_proposal/
  initial_problem_statement.md   ~1000-word problem statement (source for section 2)
  methodology_guide.md    all methodology options from the literature review
  methodology_proposals/  selection rationale + five ~1500-word proposals (P1-P5)
project_guideline/        assignment spec (PDF) and the course's original report template
literature_review/        evidence-linked literature review (see its README)
  report/                 literature_review.html, literature_review.md (generated)
  content/                review_body.html (review text), template.html (page design)
  scripts/                fetch, extract, build, verify and citation-linking scripts
  data/                   excerpts.json (41 papers, 128 verified quotes), API metadata, source texts
```

## Building the PDF

- **Overleaf:** the main document is `main.tex`.
- **Locally:** `cd overleaf && latexmk -pdf main.tex`.
- **Method versions:** to switch the method section to another version, change the `\input{sections/method}` line in `overleaf/sections/review.tex`.

## Literature review and evidence

Every claim in the literature review and in `methodology_proposal/` links to a verbatim quote. A script checked each quote against the text of the downloaded paper (128 of 128 match). To rebuild the review, or to add evidence-linked citations to a new document, see [literature_review/README.md](literature_review/README.md). In short:

```bash
literature_review/scripts/fetch_sources.sh               # once after cloning: re-download open-access papers
python3 literature_review/scripts/verify.py              # must report FAIL: 0
python3 literature_review/scripts/link_evidence.py methodology_proposal/new_file.md    # "(Ren 2025 #3)" -> checked links
python3 literature_review/scripts/add_references.py methodology_proposal/new_file.md   # append a References section
```

Requires Python 3 with `pypdf`.

**Not in the repository:** the downloaded paper PDFs and their extracted text (`literature_review/data/papers/`, `literature_review/data/text/`) are excluded by `.gitignore` because they are copyrighted. `fetch_sources.sh` re-downloads the open-access ones. Keep this repository **private**: it contains the course assignment and a student ID in `overleaf/main.tex`.

## Template

Built on the official CVPR/ICCV LaTeX author kit (<https://github.com/cvpr-org/author-kit>).
