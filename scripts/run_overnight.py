#!/usr/bin/env python3
"""Overnight run orchestrator — sequential phases, crash-safe manifest, never dies.

Runs every training + eval phase as a subprocess, tolerating failed phases:
each phase records ok/skipped/failed into results/overnight/MANIFEST.json
(rewritten after EVERY phase) and the run moves on. stdout/stderr of each phase is tee'd to
results/overnight/logs/<phase>.log and echoed live (so `tail -f overnight.out`
works under nohup).

Usage:
    python3 scripts/run_overnight.py                  # the real overnight run
    python3 scripts/run_overnight.py --smoke          # <15 min CPU plumbing check
    python3 scripts/run_overnight.py --only env_check
    python3 scripts/run_overnight.py --skip tests --force

Stdlib only. See RUNBOOK_SPARSH.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "overnight"
LOG_DIR = OUT_DIR / "logs"
MANIFEST_PATH = OUT_DIR / "MANIFEST.json"

# ---------------------------------------------------------------- timeouts (s)
TIMEOUTS = {
    "tests": 30 * 60,
    "train_stage_a": 14 * 3600,  # Stage-A at D512/L8 dominates the night
    "train_stage_b": 8 * 3600,
    "train_structure_policy": 4 * 3600,
    "eval_main_learned": 90 * 60,  # hung 3h+ on MPS once; suite saturates well before
    "eval_hard": 90 * 60,
    "eval_coupled": 90 * 60,
    "eval_dimension_scaling": 90 * 60,
    "figures": 15 * 60,
}
DEFAULT_TIMEOUT = 3 * 3600  # eval phases
SMOKE_TIMEOUT = 15 * 60  # per-phase cap in --smoke

PHASE_ORDER = [
    "env_check", "tests",
    "train_stage_a", "train_stage_b", "train_structure_policy",
    "eval_p1_learned", "eval_p1_refine", "eval_main", "eval_main_learned",
    "eval_hard",
    "eval_crossfamily", "eval_coupled", "eval_dimension_scaling",
    "eval_geometry", "eval_h2", "eval_structure_toys",
    "eval_invention", "eval_invention_heldout",
    "eval_math_coverage", "eval_cot",
    "figures", "summarize",
]

# Dirs scanned after each phase to attribute freshly written outputs.
OUTPUT_DIRS = ["results", "checkpoints", "paper/figures"]


# ------------------------------------------------------------------- utilities
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def new_outputs(since: float) -> list[str]:
    """Files under OUTPUT_DIRS modified since `since` (relative paths, capped)."""
    found = []
    for d in OUTPUT_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.stat().st_mtime >= since - 1 and not p.name.endswith(".log"):
                rel = str(p.relative_to(ROOT))
                if not rel.startswith("results/overnight/"):
                    found.append(rel)
    return found[:60]


class Manifest:
    def __init__(self, smoke: bool):
        if MANIFEST_PATH.exists():  # keep the previous run's record on rerun
            ts = datetime.fromtimestamp(
                MANIFEST_PATH.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            MANIFEST_PATH.rename(OUT_DIR / f"MANIFEST.{ts}.json")
        self.data = {
            "started_at": now_iso(),
            "finished_at": None,
            "git_commit": git_commit(),
            "smoke": smoke,
            "phases": [],
        }
        self.started_ts = time.time()

    def record(self, phase: str, cmd: str, status: str, reason: str = "",
               wall_s: float = 0.0, outputs: list[str] | None = None) -> None:
        self.data["phases"].append({
            "phase": phase, "cmd": cmd, "status": status, "reason": reason,
            "wall_s": round(wall_s, 1), "outputs": outputs or [],
        })
        self.flush()

    def flush(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(MANIFEST_PATH)

    def status(self, phase: str) -> str | None:
        for p in self.data["phases"]:
            if p["phase"] == phase:
                return p["status"]
        return None


def run_cmd(phase: str, cmd: list[str], timeout: float,
            env_extra: dict[str, str] | None = None) -> tuple[str, str]:
    """Run cmd, tee output to log + console. Returns (status, reason)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{phase}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env_extra or {})}
    # `python3 scripts/foo.py` puts scripts/ (not the repo root) on sys.path;
    # some eval scripts don't self-fix, so make `import marc` work everywhere.
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[overnight] {phase}: {shlex.join(cmd)}", flush=True)
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except OSError as e:
        log_path.write_text(f"spawn failed: {e}\n")
        return "failed", f"spawn failed: {e}"

    with open(log_path, "w") as logf:
        def tee():
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
                logf.flush()

        t = threading.Thread(target=tee, daemon=True)
        t.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            t.join(timeout=10)
            return "failed", f"timeout after {int(timeout)}s"
        t.join(timeout=10)

    if proc.returncode != 0:
        return "failed", f"exit code {proc.returncode} (see {log_path.relative_to(ROOT)})"
    return "ok", ""


# ------------------------------------------------------------------ inline phases
def phase_env_check() -> None:
    info: dict = {
        "python": sys.version,
        "platform": sys.platform,
        "time": now_iso(),
    }
    try:
        import shutil
        du = shutil.disk_usage(ROOT)
        info["disk_free_gb"] = round(du.free / 1e9, 1)
    except Exception as e:  # pragma: no cover
        info["disk_free_gb"] = f"error: {e}"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available())
    except Exception as e:
        info["torch"] = f"import failed: {e}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "env.json").write_text(json.dumps(info, indent=2))
    print(f"[overnight] env: {json.dumps(info)}", flush=True)


def pick_checkpoint(manifest: Manifest) -> Path | None:
    """Best available denoiser checkpoint for the eval battery."""
    d = ROOT / "checkpoints" / "scale_D512_L8"
    candidates = []
    if manifest.status("train_stage_b") == "ok":
        candidates.append(d / "stage_b_final.pt")
    candidates += [d / "best.pt", d / "latest.pt",
                   ROOT / "checkpoints" / "denoiser_stage_a.pt"]
    for c in candidates:
        if c.exists():
            return c
    return None


def pick_purist_checkpoint() -> Path | None:
    ck = ROOT / "checkpoints"
    if ck.exists():
        for p in sorted(ck.rglob("*purist*.pt")):
            return p
    return None


# ---------------------------------------------------------------- summarize
def _fmt(v) -> str:
    if isinstance(v, dict) and "rate" in v:
        s = f"{v['rate']:.3f}"
        ci = v.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
        return s
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (str, int, bool)):
        return str(v)
    return "—"


def _summarize_json(d: dict) -> list[str]:
    """Headline lines for the two known schemas, generic scalars otherwise."""
    lines = []
    if "splits" in d and isinstance(d["splits"], dict):  # split-eval schema
        for key in ("solver", "overall_solve_rate", "generalization_gap", "model"):
            if key in d:
                lines.append(f"- {key}: {_fmt(d[key])}")
        for split, m in d["splits"].items():
            if isinstance(m, dict):
                lines.append(
                    f"- {split}: solve_rate {_fmt(m.get('solve_rate'))}, "
                    f"pass@k {_fmt(m.get('pass_at_k'))}")
        return lines
    if "rows" in d and isinstance(d["rows"], list) and d["rows"]:  # counting schema
        for key in ("K", "test_per_family", "test_per_n", "epochs", "n_significant"):
            if key in d:
                lines.append(f"- {key}: {d[key]}")
        cols = list(d["rows"][0].keys())
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in d["rows"]:
            lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
        return lines
    for k, v in d.items():  # generic fallback
        if isinstance(v, (int, float, str, bool)) and len(str(v)) < 120:
            lines.append(f"- {k}: {_fmt(v)}")
    return lines[:20]


def phase_summarize(manifest: Manifest) -> None:
    lines = ["# Overnight run summary", ""]
    env = {}
    try:
        env = json.loads((OUT_DIR / "env.json").read_text())
    except Exception:
        pass
    total = time.time() - manifest.started_ts
    lines += [
        f"- started: {manifest.data['started_at']}",
        f"- commit: `{manifest.data['git_commit']}`",
        f"- device: {env.get('cuda_device') or ('mps' if env.get('mps_available') else 'cpu')}"
        f" (torch {env.get('torch', '?')})",
        f"- total wall time: {total/3600:.2f} h",
        f"- smoke mode: {manifest.data['smoke']}",
        "",
        "## What to look at first",
        "",
        "1. `results/p5_invention/invention.json` — menu-based structure selection (H2); "
        "`invention_heldout.json` is the held-out-pattern number.",
        "2. `results/p_coupled/coupled.json` — coupled-family scaling.",
        "3. `results/p_hard/hard_eval.json` — hard-suite headline table.",
        "",
        "## Experiments (results JSON touched by this run)",
        "",
    ]
    fresh = []
    results = ROOT / "results"
    for p in sorted(results.rglob("*.json")):
        rel = str(p.relative_to(ROOT))
        if rel.startswith("results/overnight/"):
            continue
        if p.stat().st_mtime >= manifest.started_ts - 1:
            fresh.append(p)
    if not fresh:
        lines.append("_No results JSON was written during this run._")
    for p in fresh:
        lines += [f"### {p.relative_to(ROOT)}", ""]
        try:
            lines += _summarize_json(json.loads(p.read_text()))
        except Exception as e:
            lines.append(f"- could not parse: {e}")
        lines.append("")

    lines += ["## Phase status", "",
              "| phase | status | wall (s) | reason |", "|---|---|---|---|"]
    for ph in manifest.data["phases"]:
        lines.append(f"| {ph['phase']} | {ph['status']} | {ph['wall_s']} "
                     f"| {ph['reason']} |")

    lines += ["", "## Checkpoint inventory", ""]
    ck = ROOT / "checkpoints"
    pts = sorted(ck.rglob("*.pt")) if ck.exists() else []
    if not pts:
        lines.append("_No checkpoints on disk._")
    for p in pts:
        size_mb = p.stat().st_size / 1e6
        epoch = "?"
        try:
            import torch
            d = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(d, dict):
                epoch = d.get("epoch", d.get("epochs", "?"))
        except Exception:
            pass
        lines.append(f"- `{p.relative_to(ROOT)}` — {size_mb:.1f} MB, epoch {epoch}")
    lines.append("")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[overnight] wrote {OUT_DIR / 'SUMMARY.md'}", flush=True)


# ------------------------------------------------------------------------ main
def build_phases(smoke: bool, manifest: Manifest, state: dict) -> list[dict]:
    """Ordered phase specs. cmd=None means inline. Lazy fields resolved at run.

    ``state["invention_data"]`` is filled in by the main loop after
    train_structure_policy finishes: the data source the policy actually trained
    on ("aux_required", or "toys" if the fallback fired). The invention evals
    build their cmd lazily so they eval the SAME source — eval'ing toys against
    an aux_required-trained policy (or vice versa) is a source mismatch, not a
    result.
    """
    S = smoke
    TRAIN_SP = "scripts/train_structure_policy.py"
    INV_EVAL = "scripts/run_invention_eval.py"

    def p1(solver: str, out: str) -> list[str]:
        cmd = ["python3", "scripts/run_p1_eval.py", "--solver", solver, "--out", out]
        return cmd + (["--n-id", "2", "--n-ho", "2", "--k", "2"] if S else [])

    def invention_cmd(out: str, families: list[str] | None = None) -> list[str]:
        data = state.get("invention_data") or "toys"
        cmd = ["python3", INV_EVAL,
               "--ckpt", "checkpoints/structure_policy.pt", "--out", out,
               "--data", data]
        if families:
            cmd += ["--families"] + families
        cmd += ["--eval-seeds", "2" if S else "5"]
        if S:
            cmd += ["--n", "6", "--k-refine", "2"]
        return cmd

    def heldout_skip() -> str | None:
        if state.get("invention_data") != "aux_required":
            return ("policy trained on fallback 'toys' data — no held-out "
                    "'shared' pattern to eval")
        return None

    return [
        {"name": "env_check", "inline": phase_env_check},
        {"name": "tests", "cmd": ["python3", "-m", "pytest", "-q"]},
        {"name": "train_stage_a",
         "cmd": ["python3", "scripts/train_scale.py", "--config",
                 "marc/configs/train/scale.yaml", "--stage", "a",
                 "--out-dir", "checkpoints/scale_D512_L8"]
                + (["--smoke"] if S else [])},
        {"name": "train_stage_b",
         "requires_ok": "train_stage_a",
         "cmd": ["python3", "scripts/train_scale.py", "--config",
                 "marc/configs/train/scale.yaml", "--stage", "b",
                 "--resume", "latest", "--out-dir", "checkpoints/scale_D512_L8"]
                + (["--smoke"] if S else [])},
        {"name": "train_structure_policy",
         # --exclude-family shared = the held-out-pattern protocol by default
         # (eval_invention_heldout evals the excluded pattern).
         "cmd": lambda: ["python3", TRAIN_SP,
                         "--data", "aux_required", "--epochs", "2" if S else "200",
                         "--device", "auto", "--out", "checkpoints/structure_policy.pt",
                         "--exclude-family", "shared"],
         # retried with --data toys if the aux_required run fails
         "retry_cmd": ["python3", TRAIN_SP,
                       "--data", "toys", "--epochs", "2" if S else "200",
                       "--device", "auto", "--out", "checkpoints/structure_policy.pt"]},
        {"name": "eval_p1_learned", "needs_ckpt": True,
         "cmd": p1("learned", "results/p1_baselines/metrics_learned.json")},
        {"name": "eval_p1_refine",
         "cmd": p1("refine", "results/p1_baselines/metrics.json")},
        {"name": "eval_main", "pass_ckpt": True,
         "cmd": ["python3", "scripts/run_main_eval.py"]
                + (["--n", "2", "--k", "2", "--noise-graphs", "4",
                    "--skip-ablations"] if S else [])},
        # the learned-solver row of the main table (house rule: refine above is
        # the classical row, never the system). Ablations already covered above.
        {"name": "eval_main_learned", "needs_ckpt": True,
         "cmd": ["python3", "scripts/run_main_eval.py", "--solver", "learned",
                 "--out", "results/p2_main_learned", "--skip-ablations"]
                + (["--n", "2", "--k", "2"] if S else [])},
        # wants_ckpt = soft needs_ckpt: MARC_CKPT is passed when a checkpoint
        # exists, but the phase still runs without one (these scripts self-train
        # as fallback).
        {"name": "eval_hard", "wants_ckpt": True,
         "cmd": ["python3", "scripts/run_hard_eval.py"] + (["--quick"] if S else [])},
        {"name": "eval_crossfamily", "wants_ckpt": True,
         "cmd": ["python3", "scripts/run_crossfamily_eval.py"] + (["--quick"] if S else [])},
        {"name": "eval_coupled", "wants_ckpt": True,
         "cmd": ["python3", "scripts/run_coupled_eval.py"] + (["--quick"] if S else [])},
        {"name": "eval_dimension_scaling", "wants_ckpt": True,
         "cmd": ["python3", "scripts/run_dimension_scaling.py"] + (["--quick"] if S else [])},
        {"name": "eval_geometry",
         "cmd": ["python3", "scripts/run_geometry_eval.py"]
                + (["--n", "2", "--k", "2"] if S else [])},
        {"name": "eval_h2",
         "cmd": ["python3", "scripts/run_h2_eval.py"]
                + (["--n", "2", "--k", "2", "--steps", "50"] if S else [])},
        {"name": "eval_structure_toys",
         "cmd": ["python3", "scripts/run_structure_toys.py"]},
        # both invention evals: --data matches what training actually used
        # (see build_phases docstring).
        {"name": "eval_invention",
         "requires_ok": "train_structure_policy",
         "cmd": lambda: invention_cmd("results/p5_invention/invention.json")},
        # cross-pattern generalization: eval ONLY the pattern excluded from
        # training ('shared') — the held-out-pattern number.
        {"name": "eval_invention_heldout",
         "requires_ok": "train_structure_policy", "skip_if": heldout_skip,
         "cmd": lambda: invention_cmd(
             "results/p5_invention/invention_heldout.json", families=["shared"])},
        {"name": "eval_math_coverage",
         "cmd": ["python3", "scripts/run_math_coverage.py"]},
        {"name": "eval_cot", "needs_api_key": True,
         "cmd": ["python3", "-m", "marc.eval.baselines.cot_baseline"],
         "env": {"COT_N": "2"} if S else {}},
        {"name": "figures",
         "cmds": [["python3", "scripts/plot_results.py"],
                  ["python3", "scripts/plot_hard_eval.py"]]},
        {"name": "summarize", "inline": lambda: phase_summarize(manifest)},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny args everywhere (pytest stays full); <15 min on CPU")
    ap.add_argument("--skip", default="", help="comma-separated phases to skip")
    ap.add_argument("--only", default="", help="comma-separated phases to run")
    ap.add_argument("--force", action="store_true",
                    help="keep going even if the test suite is red")
    args = ap.parse_args()

    skip = {s for s in args.skip.split(",") if s}
    only = {s for s in args.only.split(",") if s}
    for name in skip | only:
        if name not in PHASE_ORDER:
            ap.error(f"unknown phase {name!r}; known: {', '.join(PHASE_ORDER)}")

    manifest = Manifest(smoke=args.smoke)
    state: dict = {"invention_data": None}
    phases = build_phases(args.smoke, manifest, state)
    assert [p["name"] for p in phases] == PHASE_ORDER
    tests_red = False

    for spec in phases:
        name = spec["name"]
        t0 = time.time()

        def done(status: str, reason: str = "") -> None:
            cmd = spec.get("cmd") or (spec.get("cmds") or [["(inline)"]])[0]
            if "cmds" in spec:
                cmd_str = " && ".join(shlex.join(c) for c in spec["cmds"])
            elif callable(cmd):  # lazy cmd, phase gated out before resolution
                cmd_str = "(resolved at run time)"
            elif spec.get("cmd"):
                cmd_str = shlex.join(cmd)
            else:
                cmd_str = "(inline)"
            outputs = new_outputs(t0) if status == "ok" else []
            manifest.record(name, cmd_str, status, reason, time.time() - t0, outputs)
            print(f"[overnight] {name}: {status}"
                  + (f" ({reason})" if reason else ""), flush=True)

        # -- gating -------------------------------------------------------
        if only and name not in only:
            done("skipped", "not in --only")
            continue
        if name in skip:
            done("skipped", "--skip")
            continue
        # ponytail: summarize still runs on red tests — a SUMMARY.md saying
        # "aborted, tests red" beats an empty directory at 7am.
        if tests_red and name != "summarize":
            done("skipped", "tests red")
            continue
        req = spec.get("requires_ok")
        if req and manifest.status(req) != "ok":
            done("skipped", f"{req} not ok")
            continue
        skip_fn = spec.get("skip_if")
        if skip_fn:
            why = skip_fn()
            if why:
                done("skipped", why)
                continue
        if spec.get("needs_api_key") and not (
                os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            done("skipped", "no API key")
            continue

        env_extra = dict(spec.get("env", {}))
        ckpt = pick_checkpoint(manifest)
        if spec.get("needs_ckpt") and ckpt is None:
            done("skipped", "no denoiser checkpoint available")
            continue
        if (spec.get("needs_ckpt") or spec.get("pass_ckpt")
                or spec.get("wants_ckpt")) and ckpt is not None:
            env_extra["MARC_CKPT"] = str(ckpt)
        if spec.get("pass_ckpt"):
            purist = pick_purist_checkpoint()
            if purist is not None:
                env_extra["MARC_CKPT_PURIST"] = str(purist)

        timeout = TIMEOUTS.get(name, DEFAULT_TIMEOUT)
        if args.smoke:
            timeout = min(timeout, SMOKE_TIMEOUT)

        # -- execute ------------------------------------------------------
        if "inline" in spec:
            try:
                spec["inline"]()
                done("ok")
            except Exception as e:
                done("failed", f"{type(e).__name__}: {e}")
            continue

        if callable(spec.get("cmd")):  # lazy cmd — depends on earlier phases
            spec["cmd"] = spec["cmd"]()
        cmds = spec.get("cmds") or [spec["cmd"]]
        status, reasons = "ok", []
        used_cmd = cmds[-1]
        for i, cmd in enumerate(cmds):
            phase_key = name if len(cmds) == 1 else f"{name}_{i}"
            s, r = run_cmd(phase_key, cmd, timeout, env_extra)
            if s != "ok":  # still try remaining cmds (e.g. the second figure)
                status = "failed"
                reasons.append(r)
        reason = "; ".join(reasons)
        if status == "failed" and "retry_cmd" in spec:
            status, reason2 = run_cmd(f"{name}_retry", spec["retry_cmd"],
                                      timeout, env_extra)
            used_cmd = spec["retry_cmd"]
            reason = (f"first attempt failed ({reason}); retry with --data toys "
                      + ("succeeded" if status == "ok" else f"failed ({reason2})"))
        done(status, reason)
        if name == "train_structure_policy" and status == "ok":
            # which data source actually trained? the invention evals must match.
            state["invention_data"] = (
                used_cmd[used_cmd.index("--data") + 1]
                if "--data" in used_cmd else "toys")

        if name == "tests" and status == "failed":
            if args.force:
                print("[overnight] tests red but --force given; continuing", flush=True)
            else:
                tests_red = True

    manifest.data["finished_at"] = now_iso()
    manifest.flush()
    n_fail = sum(1 for p in manifest.data["phases"] if p["status"] == "failed")
    print(f"[overnight] done — {len(manifest.data['phases'])} phases, "
          f"{n_fail} failed. Manifest: {MANIFEST_PATH}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
