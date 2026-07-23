# Anonymization checklist — supplementary code artifact

Everything below runs against the **packaged copy**, never against the repo.
The repo keeps its history as-is (PROVENANCE's branch column is the recorded
provenance; see §4). Already fixed in-repo: `marc/eval/solver.py` name strings
and the `davin` solver alias; `RUNBOOK_SPARSH.md` renamed to
`RUNBOOK_OVERNIGHT.md`.

## 1. Export

```bash
git archive --format=tar --prefix=artifact/ HEAD | tar -x -C /tmp
cd /tmp/artifact
```

`git archive` drops `.git/`, untracked files, and gitignored paths
(checkpoints, local logs, `DAVIN_P2_P3_NOTES.md`). Do NOT zip the working
tree directly. On macOS, package the final tar with `COPYFILE_DISABLE=1 tar
--no-xattrs` so no `._*` metadata carrying the local username ships.

## 2. Exclude (delete from the copy)

Internal process docs — name-dense, no reviewer value:

```bash
rm -f MEETING_NOTES.md HANDOFF.md NIGHT_SESSION.md OVERNIGHT_RESULTS.md \
      OUTLINE.md FIXING_PLAN.md AAAI_READINESS.md SUMMARY.md index.html \
      CONCEPT.md TECHNICAL_GUIDE.md
rm -f paper/tex/*.fls paper/tex/*.aux paper/tex/*.log paper/tex/*.out \
      paper/tex/*.bbl paper/tex/*.blg paper/tex/*.pdf   # marc.fls holds a local /Users/<name> path
```

If CONCEPT.md / TECHNICAL_GUIDE.md should ship, scrub them instead (CONCEPT.md
line 7 has the org repo URL; TECHNICAL_GUIDE.md is clean of names).

## 3. Scrub shipped files (sed, run from the copy root)

```bash
# possessives and owner tags in comments/docstrings
grep -rlE "Quang|Sparsh|Davin|Akash" --include="*.py" --include="*.md" --include="*.yaml" . | \
xargs sed -i '' -E \
  -e "s/Quang's |Davin's |Sparsh's /the /g" \
  -e "s/ \(Davin, milestone task([^)]*)\)/ (milestone task\1)/g" \
  -e "s/\(P4, Davin\)/(P4)/g" \
  -e "s/ — Quang P4/ — P4/g" \
  -e "s/\(Akash P4 /(P4 /g" \
  -e "s/, Quang\)|, Davin\)|, Sparsh\)/)/g" \
  -e "s/Sparsh\/Davin/the team/g" \
  -e "s/\(Sparsh, MacBook M5 — /(MacBook M5 — /g"

# org URL / repo path, everywhere it survives
grep -rl "saidlaboratory" . | xargs sed -i '' \
  -e "s|https://github.com/saidlaboratory/MARC|<anonymous repo>|g" \
  -e "s|saidlaboratory/MARC|<anonymous repo>|g"

# author line + local checkpoint path
sed -i '' "s/\*\*Quang Bui, Sparsh Roy, Akash Gundimeda, Davin Yin\*\* · SAID Laboratory · July 2026/Anonymous submission · July 2026/" README.md
sed -i '' "s/SAID Laboratory//g" README.md
sed -i '' "s|/Users/sparsh/Desktop/Research/AAAI/MARC|<repo root>|" results/overnight/SUMMARY.md

# .gitignore: personal filename + tool comment
sed -i '' -e "/DAVIN_P2_P3_NOTES.md/d" \
          -e "s/# Claude Code agent worktrees.*/# agent worktrees (nested repos)/" .gitignore
```

Known hit list the pass above must cover (spot-check after):
`marc/configs/train/scale.yaml:1`, `marc/train/reward.py`,
`marc/train/rollout.py`, `marc/structure/__init__.py`,
`marc/structure/diffusion.py`, `marc/data/templates.py:145`,
`marc/refine/iterative.py:15`, `marc/eval/paper/suites.py:4`,
`marc/eval/ablations/{noise,purist,guidance}_ablation.py`,
`marc/eval/baselines/README.md:39`, `scripts/run_main_eval.py:15`,
`scripts/run_scale_experiment.py:110,144`, `scripts/run_structure_toys.py:9`,
`scripts/train_structure_pilot.py`, `results/p1_baselines/README.md:10`,
`results/p1_entrapment/report.md:45`, `results/p3_structure/pilot_report.md:6`,
`results/p4_scale/scaling_notes.md`, `results/overnight/SUMMARY.md:37`,
`README.md` (badge line 11, footer line 200).

## 4. PROVENANCE.md branch column — caveat

`paper/PROVENANCE.md` records personal branch names (`sparsh/crossover-theory`
etc.) in the commit column of rows R16–R27. **Do not rewrite these in the
repo** — they are the audit trail. In the copy only, replace each branch name
with its merge commit SHA on `main`:

```bash
# resolve each: git log --merges --oneline main | grep -F "<branch>"
# known: sparsh/fix-invention-families -> 45178c3 (#96)
#        sparsh/crossover-theory       -> 2c085e3 (#92)
sed -i '' -E "s/(sparsh|quang|davin|akash)\/[a-z0-9-]+/<sha>/g" paper/PROVENANCE.md   # then fill each <sha>
```

For branches squash-merged without a merge commit (e.g. `sparsh/fix-103-104`
if unmerged at packaging), use the squash commit SHA, or the artifact's HEAD
SHA if the work is only on HEAD.

## 5. Verify — must all come back empty

```bash
grep -rniE "quang|sparsh|davin|akash|gundimeda|imspxrsh|saidlaboratory|claude|anthropic" .
grep -rnE "\bBui\b|\bRoy\b|\bYin\b" .
grep -rn "/Users/" .
grep -rn "@gmail|@outlook|@yahoo" .
```

(`bui` alone false-positives on "build"; the word-bounded pass above is the
real check.) Then upload via an anonymizing host (anonymous.4open.science, or
Zenodo with the anonymous-record option) — not a personal or org account.
