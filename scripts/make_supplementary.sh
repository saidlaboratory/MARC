#!/usr/bin/env bash
# Build the anonymized supplementary artifact from HEAD, per paper/ANONYMIZATION.md.
# Runs entirely on an exported copy in /tmp — never touches the repo. Ends with the
# §5 verification greps; a clean run prints "ARTIFACT CLEAN", a dirty one lists every
# residual hit so the checklist gap is visible while there is still time to fix it.
#
#   scripts/make_supplementary.sh            # dry run: build + verify, report gaps
#
# Real packaging (after the paper freezes) is the same script; upload the zip via an
# anonymizing host (anonymous.4open.science / Zenodo anonymous), not a personal account.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK=/tmp/marc_artifact
rm -rf "$WORK"; mkdir -p "$WORK"

# §1 export tracked files only (drops .git, untracked, gitignored: checkpoints, logs)
git -C "$REPO" archive --format=tar --prefix=artifact/ HEAD | tar -x -C "$WORK"
cd "$WORK/artifact"

# §2 exclude internal process docs + latex build junk
rm -f MEETING_NOTES.md HANDOFF.md NIGHT_SESSION.md OVERNIGHT_RESULTS.md \
      OUTLINE.md FIXING_PLAN.md AAAI_READINESS.md SUMMARY.md index.html \
      CONCEPT.md TECHNICAL_GUIDE.md RUNBOOK_OVERNIGHT.md RUN_1.md \
      paper/ABSTRACT.md paper/CHECKLIST.md paper/ANONYMIZATION.md 2>/dev/null || true
rm -f paper/tex/*.fls paper/tex/*.aux paper/tex/*.log paper/tex/*.out \
      paper/tex/*.bbl paper/tex/*.blg paper/tex/*.pdf paper/tex/*.fdb_latexmk 2>/dev/null || true
rm -rf run_logs 2>/dev/null || true

# §3 scrub names / possessives / milestone tags / org URL / author line (BSD sed -i '')
# SPECIFIC substitutions FIRST (full author line, org, paths) — before the bare-name
# catch-all, which would otherwise mangle "Quang Bui" into "the author Bui".
[ -f README.md ] && sed -i '' \
    -e "s/\*\*Quang Bui, Sparsh Roy, Akash Gundimeda, Davin Yin\*\* · SAID Laboratory · July 2026/Anonymous submission · July 2026/" \
    -e "s/SAID Laboratory//g" README.md
grep -rl "The MARC Authors" . 2>/dev/null | xargs -r sed -i '' "s|The MARC Authors|Anonymous Authors|g"
grep -rl "saidlaboratory" . 2>/dev/null | xargs -r sed -i '' \
    -e "s|https://github.com/saidlaboratory/MARC|<anonymous repo>|g" \
    -e "s|saidlaboratory/MARC|<anonymous repo>|g"
[ -f results/overnight/SUMMARY.md ] && sed -i '' \
    "s|/Users/sparsh/Desktop/Research/AAAI/MARC|<repo root>|g" results/overnight/SUMMARY.md
grep -rl "/Users/" . 2>/dev/null | xargs -r sed -i '' -E "s|/Users/[a-z]+/[^ \"']*/MARC|<repo root>|g"
# .gitignore: personal filename, Claude tool comment, agent-worktree dir
[ -f .gitignore ] && sed -i '' \
    -e "/DAVIN_P2_P3_NOTES.md/d" \
    -e "s|# Claude Code agent worktrees.*|# agent worktrees (nested repos)|" \
    -e "s|^\.claude/|.agent-worktrees/|" .gitignore

# GENERIC name scrub LAST: tagged forms, then a bare catch-all (BSD sed: no \b, but
# these names are never substrings of legit tokens here).
grep -rlE "Quang|Sparsh|Davin|Akash" --include="*.py" --include="*.md" --include="*.yaml" \
     --include="*.txt" . 2>/dev/null | \
  xargs -r sed -i '' -E \
    -e "s/ \((Quang|Sparsh|Davin|Akash), milestone task([^)]*)\)/ (milestone task\2)/g" \
    -e "s/\((Quang|Sparsh|Davin|Akash), milestone\)/(milestone)/g" \
    -e "s/ — (Quang|Sparsh|Davin|Akash) P4/ — P4/g" \
    -e "s/\((Quang|Sparsh|Davin|Akash) P4 /(P4 /g" \
    -e "s/(Quang|Sparsh|Davin|Akash) P4/P4/g" \
    -e "s/, (Quang|Sparsh|Davin|Akash)\)/)/g" \
    -e "s/(Quang|Sparsh|Davin|Akash)\/(Quang|Sparsh|Davin|Akash)/the team/g" \
    -e "s/(Quang|Sparsh|Davin|Akash)'s/the author's/g" \
    -e "s/(Quang|Sparsh|Davin|Akash)/the author/g"
# §4 PROVENANCE branch column -> generic placeholder (resolve to SHAs before real upload)
[ -f paper/PROVENANCE.md ] && sed -i '' -E "s/(sparsh|quang|davin|akash)\/[a-z0-9-]+/<merge-sha>/g" paper/PROVENANCE.md

# a one-page quickstart lives inside the artifact
cat > SUPPLEMENTARY_README.md <<'EOF'
# MARC — supplementary code artifact (anonymous)

Reproduces every number in the paper from committed result JSONs, and documents the
full recompute path.

## Quickstart
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-lock.txt
    bash scripts/repro_all.sh verify     # import all modules, fast tests, check every cited number

`scripts/repro_all.sh rerun` prints the full regeneration path with wall-time estimates;
exact per-result commands, seeds, and data versions are in `paper/PROVENANCE.md`.
Checkpoints are omitted (regenerable from the training rows; SHA-256 manifest in PROVENANCE).
EOF

# §5 verify — everything below must come back empty
echo "== §5 verification =="
hits=0
run() { local out; out=$(eval "$1" 2>/dev/null); if [ -n "$out" ]; then echo "-- HITS: $2"; echo "$out" | head -15; hits=$((hits+1)); fi; }
run "grep -rniE 'quang|sparsh|davin|akash|gundimeda|imspxrsh|saidlaboratory|claude|anthropic' ." "names/handles/org"
run "grep -rnE '\\bBui\\b|\\bRoy\\b|\\bYin\\b' ." "surnames"
run "grep -rn '/Users/' ." "local paths"
run "grep -rnE '@gmail|@outlook|@yahoo' ." "emails"

if [ "$hits" -eq 0 ]; then
  COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/marc_supplementary.tgz -C "$WORK" artifact
  echo "ARTIFACT CLEAN -> /tmp/marc_supplementary.tgz"
else
  echo "ARTIFACT DIRTY: $hits residual categories above — fix the scrub (or the source) and re-run"
fi
exit "$hits"
