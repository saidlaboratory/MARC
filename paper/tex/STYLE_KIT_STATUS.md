# AAAI style kit status (2026-07-22)

## What landed

Official **AAAI-26 author kit** (not the 2025 stand-in). Source:
https://aaai.org/authorkit26-1/ (AuthorKit26.zip, linked from the AAAI-26
submission-instructions page). The URL serves the zip directly; no login.

Files dropped into `paper/tex/`:

- `aaai2026.sty` — `ProvidesPackage{aaai2026}[2026/06/17 AAAI 2026 Submission format]`
- `aaai2026.bst`

Both taken from `AuthorKit26/AnonymousSubmission/LaTeX/`; byte-identical to the
CameraReady copies, so no swap needed at camera-ready time.

Verified: a 10-line hello-world with `\usepackage[submission]{aaai2026}` compiles
clean under pdflatex (TeX Live 2025), 1-page PDF.

## Preamble diff for marc_aaai.tex

Whoever owns marc_aaai.tex applies this (matches the TODO in its header, lines 1-11):

1. Replace line 13 `\documentclass[10pt,letterpaper,twocolumn]{article}` with
   `\documentclass[letterpaper]{article}` + `\usepackage[submission]{aaai2026}`.
2. Delete lines 14-15 (`geometry`, `\columnsep`) and line 24 (`hyperref`) — both
   packages are on the kit's forbidden list. Add `\urlstyle{rm}` / `\def\UrlFont{\rm}`
   if url styling shifts.
3. Line 654: `\bibliographystyle{plainnat}` -> `\bibliographystyle{aaai2026}`.

Kit notes: `natbib` must be loaded with **no options** (line 23 is already fine);
`\usepackage{caption}` is required by the kit template if captions are customized;
author block for submission stays anonymous (the `submission` option hides it).
Reference template: `AuthorKit26/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.tex`
(kit zip extracted at /tmp/aaai_kit/ if still around; otherwise re-download from the URL above).

## AAAI-27 migration (2026-07-29)

Source of the kit: `AuthorKit27.zip` supplied by the authors. `aaai2027.sty` and
`aaai2027.bst` are copied into this directory unmodified.

Changes forced by the 2027 kit, each of which the AAAI-26 draft violated:

- **Font packages are forbidden.** The kit loads newtxtext/helvet/courier itself and says
  "DO NOT add \usepackage{times}, \usepackage{helvet}, \usepackage{courier}, or any other
  font package." The 2026 preamble loaded all three, plus a XeTeX `fontspec` shim. All
  removed.
- **`\clearpage` is on the disallowed-commands list** ("No page breaks of any kind"). The
  appendix used it. Removed; the one-column switch alone now starts the supplementary block.
- **Section order is mandated**: main content, content appendices, ethical statement,
  acknowledgments, references, supplementary material. The "Use of AI Assistants" statement
  had been placed after the bibliography and is now before it.
- `\pdfinfo{/TemplateVersion (2027.1)}`, `\urlstyle{rm}` and `\def\UrlFont{\rm}` added per
  the template.

**This directory can no longer be built locally.** `aaai2027.sty` hard-requires pdfTeX
("pdfTeX is required to compile this document") and aborts under XeTeX, which is the only
engine `tectonic` provides. Build on Overleaf with pdfLaTeX, which is where the submission
artifact should come from anyway.

For local pagination estimates only, copy `paper/tex/` to a scratch directory and swap
`aaai2027` for `aaai2026` plus the old font block; the two styles share the two-column
letterpaper geometry and metrically equivalent Times clones, so the estimate is close but
is **not** the submission artifact.

**Open:** the 2027 kit does not state a numeric page limit ("Check the conference's
instructions in their website"). Under the 2026-proxy build the content occupies 7 pages
and the AI-use statement spills roughly four lines onto an eighth. Confirm the AAAI-27
limit before submitting; if it is 7, cut four lines or move the statement after the
references.
