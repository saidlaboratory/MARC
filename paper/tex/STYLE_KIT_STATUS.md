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
