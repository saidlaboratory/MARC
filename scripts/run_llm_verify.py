"""Verified reasoning: does LLM-formalize-then-solve beat the LLM solving directly?

Two pipelines on a MATH-500 sample:
  * direct    — the LLM (Gemini) solves via chain-of-thought; we extract its final answer.
  * formalize — the LLM converts the problem into a system of equations + a query variable
                (MARC's autoformalization target); we solve it *exactly* with SymPy and read
                off the query. MARC's role is the formalization frame + exact symbolic solve
                (the diffusion solver is numeric and cannot produce MATH's exact answers).

Answer equality is checked symbolically (SymPy), so 0.15 == 3/20, etc. Reports accuracy of
each pipeline, the formalization coverage, and — the question that matters — whether exact
solving *fixes* the LLM's computational mistakes (formalize > direct).

Honest scope: this is verified/tool-augmented reasoning (LLM + CAS), a known paradigm; the
point is the empirical delta, reported straight. Needs GEMINI_API_KEY.

Run:  GEMINI_API_KEY=... python scripts/run_llm_verify.py [--n 25]
Writes results/p_llm/verify.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import sympy as sp

DATA = Path("marc/data/math_benchmark/math500_sample.jsonl")
CACHE = Path("results/p_llm/llm_cache.json")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _client():
    import openai
    return openai.OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE)


_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE.read_text())
        except (OSError, ValueError):
            _cache = {}
    return _cache


def call(prompt: str, model="gemini-flash-lite-latest", tag="") -> str:
    c = _load_cache()
    key = f"{model}::{tag}::{prompt}"
    if key in c:
        return c[key]
    client = _client()
    delay = 4.0
    import openai
    for _ in range(6):
        try:
            r = client.chat.completions.create(model=model, max_completion_tokens=1400,
                                               messages=[{"role": "user", "content": prompt}])
            out = r.choices[0].message.content
            c[key] = out
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(c))
            return out
        except openai.RateLimitError:
            time.sleep(delay); delay *= 2
    raise RuntimeError("rate limited")


def sym_equal(pred: str, gold: str) -> bool:
    """Symbolic equality of two answer strings (handles fractions, radicals, decimals)."""
    def norm(s):
        s = s.strip().strip("$").replace("\\", "").replace("%", "").replace(" ", "")
        s = s.replace("pi", "pi")
        try:
            return sp.nsimplify(sp.sympify(s, rational=True), rational=False)
        except Exception:
            try:
                return sp.sympify(s)
            except Exception:
                return None
    a, b = norm(pred), norm(gold)
    if a is None or b is None:
        return pred.strip().strip("$").replace(" ", "") == gold.strip().strip("$").replace(" ", "")
    try:
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return a == b


def direct_answer(problem: str) -> str:
    p = (f"Solve this problem. Reason briefly, then give ONLY the final answer on the last "
         f"line in the form: ANSWER: <value>\n\n{problem}")
    txt = call(p, tag="direct")
    m = re.search(r"ANSWER:\s*(.+)", txt)
    return (m.group(1).strip() if m else txt.strip().splitlines()[-1].strip())


def formalize_and_solve(problem: str):
    """Returns (coverage: bool, answer: str|None)."""
    p = (
        "Convert this math problem into equations and solve symbolically. Output ONLY compact "
        "JSON: {\"vars\": [..], \"equations\": [\"lhs = rhs\", ..], \"query\": \"expression to "
        "evaluate\"}. Use ^ for powers, standard function names (sqrt, sin, ...). If it cannot "
        "be expressed as equations in variables, output {\"vars\":[],\"equations\":[],\"query\":\"\"}.\n\n"
        f"Problem: {problem}"
    )
    txt = call(p, tag="formalize")
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return False, None
    try:
        spec = json.loads(m.group(0))
    except ValueError:
        return False, None
    eqs = spec.get("equations") or []
    query = (spec.get("query") or "").strip()
    if not eqs or not query:
        return False, None
    try:
        syms = {v: sp.Symbol(v) for v in spec.get("vars", [])}
        # also pull any symbols appearing in equations/query
        sp_eqs = []
        for e in eqs:
            if "=" not in e:
                continue
            l, r = e.split("=", 1)
            sp_eqs.append(sp.Eq(sp.sympify(l.replace("^", "**"), locals=syms),
                                sp.sympify(r.replace("^", "**"), locals=syms)))
        sol = sp.solve(sp_eqs, list(syms.values()), dict=True)
        if not sol:
            return True, None
        q = sp.sympify(query.replace("^", "**"), locals=syms)
        for s in sol:  # first solution that evaluates to a number
            val = q.subs(s)
            if val.free_symbols == set():
                return True, str(sp.nsimplify(val))
        return True, None
    except Exception:
        return True, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    problems = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()][: args.n]

    direct_ok = form_ok = covered = 0
    rows = []
    for i, p in enumerate(problems):
        gold = p["answer"]
        da = direct_answer(p["problem"])
        d_correct = sym_equal(da, gold)
        cov, fa = formalize_and_solve(p["problem"])
        f_correct = bool(fa) and sym_equal(fa, gold)
        direct_ok += d_correct
        covered += cov
        form_ok += f_correct
        rows.append({"type": p["type"], "level": p["level"], "gold": gold,
                     "direct": da, "direct_correct": d_correct,
                     "formalized": cov, "formalize_answer": fa, "formalize_correct": f_correct})
        print(f"[{i+1}/{len(problems)}] {p['type'][:12]:12s} gold={gold[:14]:14s} "
              f"direct={'Y' if d_correct else 'n'} formalize={'Y' if f_correct else ('cov' if cov else '-')}",
              flush=True)

    n = len(problems)
    print(f"\ndirect accuracy      : {direct_ok}/{n} = {direct_ok/n:.3f}")
    print(f"formalize coverage   : {covered}/{n} = {covered/n:.3f}")
    print(f"formalize accuracy   : {form_ok}/{n} = {form_ok/n:.3f}")
    denom = covered if covered else 1
    print(f"formalize acc | covered: {form_ok}/{covered} = {form_ok/denom:.3f}")
    out = Path("results/p_llm"); out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps({
        "n": n, "model": "gemini-flash-lite-latest",
        "direct_accuracy": direct_ok / n, "formalize_coverage": covered / n,
        "formalize_accuracy": form_ok / n, "rows": rows}, indent=2))
    print(f"wrote {out/'verify.json'}")


if __name__ == "__main__":
    main()
