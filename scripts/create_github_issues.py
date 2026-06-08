#!/usr/bin/env python3
"""Create GitHub issues from task definitions in index.html."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = "saidlaboratory/MARC"
ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"

OWNER_TO_LOGIN = {
    "Davin": "dyin08",
    "Quang": "duckyquang",
    "Akash": "Gun-Akash",
    "Sparsh": "ImSpxrsh",
}

PHASE_LABELS = {
    "P0": "P0 · infrastructure",
    "P1": "P1 · value-diffusion",
    "P2": "P2 · checker RL",
    "P3": "P3 · structure diffusion",
    "P4": "P4 · scope & scale",
}


def html_to_md(html: str) -> str:
    s = html.strip()
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    s = re.sub(r"<b>(.*?)</b>", r"**\1**", s, flags=re.S)
    s = re.sub(r"<em>(.*?)</em>", r"*\1*", s, flags=re.S)
    s = re.sub(r"<ol class=\"tsteps\">", "\n", s)
    s = re.sub(r"</ol>", "\n", s)
    s = re.sub(r"<li>(.*?)</li>", r"1. \1\n", s, flags=re.S)
    s = re.sub(r'<p class="tl">(.*?)</p>', r"\1\n\n", s, flags=re.S)
    s = re.sub(
        r'<p class="tbox (?:done|push|parallel)">(.*?)</p>',
        r"> \1\n\n",
        s,
        flags=re.S,
    )
    s = re.sub(r"<[^>]+>", "", s)
    return (
        s.replace("&amp;", "&")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("\n\n\n", "\n\n")
        .strip()
    )


def parse_tasks(text: str) -> list[dict]:
    block = text.split("const tasks=[", 1)[1].split("];", 1)[0]
    pattern = re.compile(
        r"\{t:'((?:\\'|[^'])*)',p:'([^']*)',o:\[(.*?)\],m:(true|false),d:`((?:[^`\\]|\\.)*)`\}",
        re.S,
    )
    tasks = []
    for m in pattern.finditer(block):
        title = m.group(1).replace("\\'", "'")
        phase = m.group(2)
        owners = [
            o.strip().strip("'")
            for o in m.group(3).split(",")
            if o.strip()
        ]
        math = m.group(4) == "true"
        body_html = m.group(5)
        tasks.append(
            {
                "title": title,
                "phase": phase,
                "owners": owners,
                "math": math,
                "body": html_to_md(body_html),
            }
        )
    return tasks


class GitHub:
    def __init__(self, token: str) -> None:
        self.token = token

    def _request(self, method: str, path: str, data: dict | None = None) -> dict | list:
        url = f"https://api.github.com/repos/{REPO}{path}"
        cmd = [
            "curl", "-sS", "-X", method, url,
            "-H", f"Authorization: token {self.token}",
            "-H", "Accept: application/vnd.github+json",
            "-H", "Content-Type: application/json",
        ]
        if data is not None:
            cmd += ["-d", json.dumps(data)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{method} {path} curl failed: {result.stderr}")
        raw = result.stdout.strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("message") and method == "POST" and path != "/labels":
            raise RuntimeError(f"{method} {path} failed: {parsed}")
        return parsed

    def ensure_label(self, name: str, color: str, description: str) -> None:
        parsed = self._request(
            "POST",
            "/labels",
            {"name": name, "color": color, "description": description},
        )
        if isinstance(parsed, dict) and parsed.get("message") == "Validation Failed":
            print(f"  · label exists: {name}")
        else:
            print(f"  + label: {name}")

    def create_issue(self, task: dict) -> str:
        labels = [PHASE_LABELS[task["phase"]]]
        if task["math"]:
            labels.append("math-heavy")
        labels.append("parallel-track")

        assignees = [
            OWNER_TO_LOGIN[o]
            for o in task["owners"]
            if o in OWNER_TO_LOGIN
        ]

        body = (
            f"**Phase:** {task['phase']}\n"
            f"**Owner(s):** {', '.join(task['owners'])}\n\n"
            f"---\n\n"
            f"{task['body']}\n\n"
            f"---\n\n"
            f"_Auto-created from the [MARC task board](https://github.com/{REPO}) "
            f"(`index.html`). Update the board if scope changes._"
        )

        payload = {
            "title": task["title"],
            "body": body,
            "labels": labels,
        }
        if assignees:
            payload["assignees"] = assignees

        issue = self._request("POST", "/issues", payload)
        url = issue["html_url"]
        print(f"  ✓ #{issue['number']} {task['title']} → {url}")
        return url


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("Set GITHUB_TOKEN or GH_TOKEN.", file=sys.stderr)
        return 1

    tasks = parse_tasks(INDEX.read_text())
    if not tasks:
        print("No tasks parsed from index.html", file=sys.stderr)
        return 1

    gh = GitHub(token)
    print(f"Creating labels…")
    label_defs = [
        ("P0 · infrastructure", "1d76db", "Phase 0 — parallel infrastructure tracks"),
        ("P1 · value-diffusion", "f0a830", "Phase 1 — value-diffusion MVP"),
        ("P2 · checker RL", "e35d6b", "Phase 2 — checker fine-tuning / GRPO"),
        ("P3 · structure diffusion", "0e8a16", "Phase 3 — structure diffusion (preliminary)"),
        ("P4 · scope & scale", "5319e7", "Phase 4 — post-submission scope & scale"),
        ("math-heavy", "b60205", "Math-heavy modeling task"),
        ("parallel-track", "fbca04", "Can start in parallel — no blocking wait"),
    ]
    for name, color, desc in label_defs:
        gh.ensure_label(name, color, desc)

    print(f"\nCreating {len(tasks)} issues…")
    urls = []
    for task in tasks:
        urls.append(gh.create_issue(task))

    print(f"\nDone — {len(urls)} issues created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
