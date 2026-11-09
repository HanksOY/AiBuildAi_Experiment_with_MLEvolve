"""Inspector agent: checks each dashboard agent's result before the user approves it.

Two layers, both advisory (the Inspector warns, it never blocks):
  1. Rule checks: plain code, no LLM, no network. Pure functions of the project state, except
     `sniff` (reads the first bytes of the data files).
  2. A Claude check: one LLM call per inspected result. It compares the result with the data, the
     user's messages and the earlier steps, and reports contradictions and gaps.

A check is {"label", "ok": True|False, "detail", "src": "Rule"|"Claude", "blocking": bool}.
blocking=True marks a problem (red); False with ok=False is a warning. Nothing here stops a step.

Stages: setup (rules only), data, task, eda, plan. Each result is identified by a digest of its
content, so a check that finishes after the result changed is dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

STAGES = ("setup", "data", "task", "eda", "plan")
STEPPER = {"setup": "configuration", "data": "data", "task": "task", "eda": "eda", "plan": "plan"}
# Who receives "Send the warnings": data notes go into the task text, the rest to the agent that made the result.
TARGET = {"setup": "setup", "data": "task", "task": "task", "eda": "eda", "plan": "plan"}
WHAT = {"setup": "the settings", "data": "the Data agent's result", "task": "the task draft",
        "eda": "the EDA report and data plan", "plan": "the training plan"}

# Planning estimates for the budget check (stated to the user with the result).
DRAFT_WRITE_S = 300       # a Claude-written first draft: about 5 min
STEP_WRITE_S = 240        # a later improve/debug step: about 4 min
RUN_S_PER_100MB = 60      # running a script: about 1 min per 100 MB of data, at least 1 min
BIG_FILE_BYTES = 500 * 1024 * 1024

NON_DATA = {".doc", ".docx", ".pdf", ".ppt", ".pptx", ".rtf", ".odt", ".pages", ".key"}
JUNK = re.compile(r"^(~\$|\.~lock|\._)|^(thumbs\.db|desktop\.ini|\.ds_store)$", re.I)
ANSWER_NAME = re.compile(r"(solution|answer|label|ground[_ -]?truth|test[_ -]?(y|target|labels?))", re.I)
METRIC_RE = re.compile(r"\b(auc|roc|pr[- ]?auc|accuracy|f1|recall|precision|rmse|mse|mae|r2|r²|log ?loss|"
                       r"cross[- ]entropy|kappa|map@?\d*|ndcg|dice|iou|mcc|brier|spearman|pearson)\b", re.I)


def check(label: str, ok: bool, detail: str = "", src: str = "Rule", blocking: bool = False) -> dict:
    return {"label": label, "ok": bool(ok), "detail": detail, "src": src, "blocking": bool(blocking and not ok)}


def digest(stage: str, d: dict) -> str:
    """Identity of the result being inspected; a later result of the same stage has another digest."""
    c = d.get("config", {})
    part = {
        "setup": lambda: c,
        "data": lambda: d.get("data", {}).get("profile"),
        "task": lambda: d.get("task", {}).get("draft"),
        "eda": lambda: [d.get("eda", {}).get("report"), d.get("eda", {}).get("data_plan")],
        "plan": lambda: [d.get("plan", {}).get("plan"), c],
    }[stage]()
    return hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── file facts ──────────────────────────────────────────────────────────────


def sniff(path: Path) -> str:
    """'csv' | 'xls' | 'zip' (xlsx, docx) | 'parquet' | 'binary' | 'unknown', from the first bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(2048)
    except OSError:
        return "unknown"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "xls"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"PAR1"):
        return "parquet"
    if not head:
        return "unknown"
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    first = text.splitlines()[0] if text.splitlines() else ""
    return "csv" if ("," in first or "\t" in first or ";" in first) else "text"


EXPECT = {".csv": {"csv", "text"}, ".tsv": {"csv", "text"}, ".txt": {"csv", "text"}, ".xls": {"xls"},
          ".xlsx": {"zip"}, ".parquet": {"parquet"}, ".json": {"text", "csv"}, ".jsonl": {"text", "csv"}}
READ_AS = {"csv": "`pd.read_csv`", "xls": "`pd.read_excel` (xlrd)", "zip": "`pd.read_excel` (openpyxl)",
           "parquet": "`pd.read_parquet`"}


def file_facts(profile: dict, local_dir: Path | None) -> list[dict]:
    """[{name, size, ext, kind}] for the data files; kind is None when the folder is not readable here."""
    out = []
    for f in profile.get("files", []):
        name = str(f.get("name", ""))
        kind = sniff(local_dir / name) if local_dir is not None else None
        out.append({"name": name, "size": int(f.get("size") or 0), "ext": Path(name).suffix.lower(), "kind": kind})
    return out


# ── rule checks ─────────────────────────────────────────────────────────────


def setup_rules(d: dict, cpus: int | None = None) -> list[dict]:
    c = d.get("config", {})
    steps, limit_s = int(c.get("steps", 1) or 1), float(c.get("time_limit_h", 1) or 1) * 3600
    out = [check("Time per step", limit_s / steps >= 120,
                 f"{limit_s / steps:.0f} s per step ({steps} steps in {c.get('time_limit_h')} h). One Claude-written "
                 "step takes about 4–5 min, so most steps will not get their turn." if limit_s / steps < 120 else
                 f"{limit_s / steps / 60:.0f} min per step")]
    exec_s = float(c.get("exec_timeout_min", 60) or 60) * 60
    out.append(check("Script time limit fits the budget", exec_s <= limit_s,
                     f"One script may run {exec_s / 60:.0f} min, longer than the whole budget "
                     f"({limit_s / 60:.0f} min)." if exec_s > limit_s else f"{exec_s / 60:.0f} min per script"))
    if cpus:
        out.append(check("Parallel solutions fit the CPU", int(c.get("parallel", 1)) <= cpus,
                         f"{c.get('parallel')} parallel solutions on {cpus} CPU cores"))
    return out


def data_rules(d: dict, facts: list[dict]) -> list[dict]:
    prof = d.get("data", {}).get("profile") or {}
    out = [check("Data folder is readable", bool(prof.get("files")) and not prof.get("error"),
                 prof.get("error") or f"{len(prof.get('files', []))} files")]
    answers = [f["name"] for f in facts if ANSWER_NAME.search(Path(f["name"]).stem)]
    out.append(check("No answer or label file", not answers,
                     f"{', '.join(f'`{a}`' for a in answers)} looks like a test-label file. The agents must never "
                     "read test labels; move it out of the folder." if answers else "none found"))
    wrong = [f for f in facts if f["kind"] and f["ext"] in EXPECT and f["kind"] not in EXPECT[f["ext"]]]
    out.append(check("File type matches its name", not wrong,
                     " ".join(f"`{f['name']}` is really a {f['kind'].upper()} file. Read it with "
                              f"{READ_AS.get(f['kind'], 'a matching reader')}." for f in wrong)
                     if wrong else "checked from the first bytes of each file"))
    junk = [f["name"] for f in facts if JUNK.search(f["name"]) or f["ext"] in NON_DATA]
    out.append(check("Only data files in the folder", not junk,
                     f"{', '.join(f'`{j}`' for j in junk)} {'is' if len(junk) == 1 else 'are'} not data "
                     "(lock files or documents). Move them out of the folder." if junk else ""))
    tables = [t for t in prof.get("tables", []) if t.get("role") == "train"]
    target = next((t.get("target_guess") for t in tables if t.get("target_guess")), None)
    out.append(check("Target column found", bool(target), f"`{target}`" if target else
                     "No table has an obvious target column. Name it in your message to the Task agent."))
    for t in tables:
        counts = t.get("target_counts") or {}
        if t.get("target_guess") and len(counts) == 1:
            value = next(iter(counts))
            out.append(check(f"Target has 2 or more values in {t['name']}", False,
                             f"`{t['target_guess']}` is {value} in all {t.get('rows', '?')} rows of `{t['name']}`. "
                             "The other class must come from another file; the task must say how to combine them."))
        rows = int(t.get("rows") or 0)
        if 0 < rows < 50:
            out.append(check(f"Enough rows in {t['name']}", False, f"only {rows} rows: scores will be noisy"))
        cells = rows * int(t.get("columns") or 0)
        if cells and int(t.get("missing_cells") or 0) / cells > 0.3:
            out.append(check(f"Missing values in {t['name']}", False,
                             f"{int(t['missing_cells']) / cells:.0%} of the cells are empty"))
    return out


def task_rules(d: dict) -> list[dict]:
    text = d.get("task", {}).get("draft") or ""
    metric = METRIC_RE.search(text)
    return [check("Metric named", bool(metric), f"`{metric.group(0)}`" if metric else "The task names no metric."),
            check("Score line required", "Final Validation Score" in text,
                  "`Final Validation Score: <score>`" if "Final Validation Score" in text else
                  "The task must ask each script to print `Final Validation Score: <score>`; without it no solution "
                  "gets a score.", blocking=True),
            check("Validation method named", bool(re.search(r"\b(cross[- ]validation|k-?fold|\d+-fold|hold-?out|"
                                                            r"validation split|train/val)", text, re.I)),
                  "" if re.search(r"fold|hold-?out|validation split", text, re.I) else "Say how to validate "
                  "(for example stratified 5-fold CV)."),
            check("Outputs go to ./working/", "working" in text, "" if "working" in text else
                  "Name the output files and put them in `./working/`.")]


def eda_rules(d: dict) -> list[dict]:
    eda = d.get("eda", {})
    report, plan = eda.get("report") or "", eda.get("data_plan") or ""
    tables = [t["name"] for t in (d.get("data", {}).get("profile") or {}).get("tables", []) if t.get("role") == "train"]
    missed = [t for t in tables if t not in report and Path(t).stem not in report]
    return [check("Report covers every data file", not missed,
                  f"The report does not mention {', '.join(f'`{m}`' for m in missed)}." if missed else
                  f"{len(tables)} of {len(tables)} data files"),
            check("Data plan written", len(plan.strip()) > 40, "" if len(plan.strip()) > 40 else "The data plan is empty."),
            check("No test labels in the plan", not re.search(r"\btest (labels?|answers?)\b", plan, re.I)
                  or bool(re.search(r"(never|not|no) [^.]{0,40}test (labels?|answers?)", plan, re.I)), "")]


def budget(d: dict, facts: list[dict]) -> tuple[float, str]:
    """(estimated seconds, how it was estimated) for the configured run."""
    c = d.get("config", {})
    steps, drafts = int(c.get("steps", 1)), min(int(c.get("initial_drafts", 1)), int(c.get("steps", 1)))
    par = max(1, int(c.get("parallel", 1)))
    mb = sum(f["size"] for f in facts) / 1e6
    run_s = max(60.0, RUN_S_PER_100MB * mb / 100)
    est = drafts * (DRAFT_WRITE_S + run_s) + max(0, steps - drafts) * (STEP_WRITE_S + run_s) / par
    how = (f"{drafts} first drafts one at a time (about {DRAFT_WRITE_S // 60} min to write each), then "
           f"{max(0, steps - drafts)} steps {par} at a time (about {STEP_WRITE_S // 60} min to write each); about "
           f"{run_s / 60:.0f} min to run each script on {mb:.1f} MB of data" if mb < 10 else
           f"{run_s / 60:.0f} min to run each script on {mb:.0f} MB of data")
    return est, how


def plan_rules(d: dict, facts: list[dict]) -> tuple[list[dict], dict | None]:
    """(checks, fix) — fix is a settings change the user can apply with one button."""
    c = d.get("config", {})
    limit_s = float(c.get("time_limit_h", 1)) * 3600
    est, how = budget(d, facts)
    fits = est <= limit_s
    need_h = math.ceil(est / 1800) / 2  # round up to half an hour
    out = [check("Budget fits the plan", fits,
                 f"Estimate {est / 3600:.1f} h, limit {c.get('time_limit_h')} h: {how}." if not fits else
                 f"estimate {est / 3600:.1f} h of {c.get('time_limit_h')} h", blocking=est > 1.25 * limit_s)]
    big = [f for f in facts if f["size"] >= BIG_FILE_BYTES]
    if big and int(c.get("parallel", 1)) > 1:
        out.append(check("Memory for parallel scripts", False,
                         f"{c.get('parallel')} scripts at once each load `{big[0]['name']}` "
                         f"({big[0]['size'] / 1e6:.0f} MB). Read it in chunks or lower the parallel solutions."))
    out.append(check("Plan written", len((d.get("plan", {}).get("plan") or "").strip()) > 40, ""))
    fix = None if fits else {"label": f"Set the time limit to {need_h:g} h", "changes": {"time_limit_h": need_h}}
    return out, fix


# ── Claude check ────────────────────────────────────────────────────────────

LLM_SYSTEM = """You are the Inspector agent of the MLEvolve dashboard. Another agent produced the result below. Check
it against the facts: the data profile, what the user said, and the earlier approved steps.

How MLEvolve works, so you do not report normal things:
- The search writes and tries many solutions and model families itself. A task or plan need not name models.
- A folder with training data only is normal: each solution validates on its own split of the training data. No test
  file and no submission file are needed.
- Documents and lock files in the data folder are ignored by the run; the rule checks already report them.
- Small data is the user's data, not an error. Cross-validation already handles it.

Report a failed check only when the result would:
- make the run fail, or read or combine the data wrongly (wrong target, wrong files, wrong format);
- leak information between training and validation, or use test labels;
- contradict what the user asked or an earlier approved step;
- leave out something the user asked for.
Severity "problem" only when the run would fail or its score would be invalid; else "warning". Do not repeat the rule
checks listed. No style or wording advice. Report at most 4 checks. Passed checks need no detail. Plain English, one
short sentence per detail."""


def llm_prompt(stage: str, d: dict, said: list[str], rules: list[dict]) -> str:
    prof = d.get("data", {}).get("profile") or {}
    slim = {"files": prof.get("files"), "tables": [{k: t.get(k) for k in ("name", "role", "rows", "columns",
                                                                          "missing_cells", "target_guess",
                                                                          "target_counts")}
                                                   for t in prof.get("tables", [])]}
    result = {
        "data": lambda: json.dumps(prof)[:8000],
        "task": lambda: d.get("task", {}).get("draft", ""),
        "eda": lambda: (d.get("eda", {}).get("report") or "")[:8000] + "\n\n## Data plan\n" +
                       (d.get("eda", {}).get("data_plan") or ""),
        "plan": lambda: (d.get("plan", {}).get("plan") or "") + "\n\nSettings: " + json.dumps(d.get("config", {})),
    }[stage]()
    earlier = []
    if stage in ("eda", "plan"):
        earlier.append("## Accepted task\n" + (d.get("task", {}).get("draft") or "")[:5000])
    if stage == "plan":
        earlier.append("## Approved data plan\n" + (d.get("eda", {}).get("data_plan") or "(EDA skipped)")[:4000])
        qa = zip(d.get("plan", {}).get("questions", []), d.get("plan", {}).get("answers", []))
        earlier.append("## The user's answers to the plan questions\n" +
                       "\n".join(f"- {q.get('question')}: {a}" for q, a in qa))
    rules_txt = "\n".join(f"- {r['label']}: {'ok' if r['ok'] else 'FAIL'} {r['detail']}" for r in rules)
    return (f"{LLM_SYSTEM}\n\n# Result to check: {WHAT[stage]}\n{result}\n\n# Data profile\n{json.dumps(slim)[:6000]}\n\n"
            + "\n\n".join(earlier) + "\n\n# What the user said\n" + ("\n".join(f"- {s[:500]}" for s in said) or "(nothing)")
            + f"\n\n# Rule checks already done\n{rules_txt}\n\n"
            'Answer with JSON only: {"checks": [{"label": "<short check name>", "ok": true|false, '
            '"detail": "<one sentence; empty when ok>", "severity": "warning" or "problem"}]}')


def parse_llm(answer: dict) -> list[dict]:
    out = []
    for c in (answer.get("checks") or [])[:5]:
        if not isinstance(c, dict) or not str(c.get("label", "")).strip():
            continue
        ok = c.get("ok") is True
        out.append(check(str(c["label"])[:80], ok, str(c.get("detail") or "")[:400], "Claude",
                         blocking=(not ok and c.get("severity") == "problem")))
    return out


# ── message ─────────────────────────────────────────────────────────────────


def summary(stage: str, checks: list[dict], llm_note: str | None = None) -> str:
    bad = [c for c in checks if not c["ok"] and c["blocking"]]
    warn = [c for c in checks if not c["ok"] and not c["blocking"]]
    passed = len(checks) - len(bad) - len(warn)
    parts = [f"{passed} passed"] + ([f"{len(warn)} warning{'s' * (len(warn) != 1)}"] if warn else []) + \
        ([f"{len(bad)} problem{'s' * (len(bad) != 1)}"] if bad else [])
    text = f"I checked {WHAT[stage]}: **{', '.join(parts)}.**"
    if bad or warn:
        text += " The Inspector only warns; you decide."
    if llm_note:
        text += f"\n\n_{llm_note}_"
    return text


def note(checks: list[dict]) -> str:
    """Short text for the stage bar, or ''."""
    bad = sum(1 for c in checks if not c["ok"] and c["blocking"])
    warn = sum(1 for c in checks if not c["ok"] and not c["blocking"])
    if bad:
        return f"{bad} problem{'s' * (bad != 1)} found"
    if warn:
        return f"{warn} warning{'s' * (warn != 1)}"
    return ""


def send_text(stage: str, checks: list[dict]) -> str:
    items = "\n".join(f"- {c['label']}: {c['detail']}" for c in checks if not c["ok"])
    if stage == "data":
        return f"Add these notes about the data to the task, so every agent in the run knows them:\n{items}"
    return f"The Inspector found these issues. Fix them, or tell me why not:\n{items}"
