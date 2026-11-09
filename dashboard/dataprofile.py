"""Profile a data folder with the standard library only.

    python dashboard/dataprofile.py <folder>   prints the profile as JSON

The Data agent uses it directly when the folder is readable from the dashboard
process, and runs this file inside WSL otherwise. Files that look like test or
answer files get only their row and column counts: no target statistics, so
no test labels are read.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

MAX_FILES = 60
MAX_TABLES = 6
SCAN_ROWS = 200_000
TARGET_NAMES = ("target", "label", "labels", "y", "class", "outcome", "diagnosis", "survived", "is_fraud",
                "default", "churn", "response")
NO_STATS_HINTS = ("test", "solution", "answer", "submission", "holdout")
TABLE_SUFFIXES = {".csv", ".tsv"}


def profile(folder: str | Path) -> dict:
    root = Path(folder)
    if not root.exists():
        return {"path": str(folder), "error": "folder not found"}
    if root.is_file():
        files, root = [root], root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.is_file())
    out = {"path": str(folder), "n_files": len(files), "files": [], "tables": []}
    for p in files[:MAX_FILES]:
        out["files"].append({"name": p.relative_to(root).as_posix(), "size": p.stat().st_size})
    tables = [p for p in files if p.suffix.lower() in TABLE_SUFFIXES][:MAX_TABLES]
    for p in tables:
        out["tables"].append(_table(p, root))
    return out


def _table(path: Path, root: Path) -> dict:
    name = path.relative_to(root).as_posix()
    stats = not any(h in path.name.lower() for h in NO_STATS_HINTS)
    info = {"name": name, "role": "train" if stats else "test or answer file (counts only)"}
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            header = next(reader, [])
            info["columns"] = len(header)
            info["column_names"] = header[:60]
            target = _guess_target(header) if stats else None
            t_idx = header.index(target) if target else None
            rows, missing, counts, many = 0, 0, {}, False
            for row in reader:
                rows += 1
                if rows <= SCAN_ROWS and stats:
                    missing += sum(1 for c in row if c.strip() in ("", "NA", "NaN", "nan", "null"))
                    if t_idx is not None and t_idx < len(row) and not many:
                        v = row[t_idx]
                        counts[v] = counts.get(v, 0) + 1
                        many = len(counts) > 20
            info["rows"] = rows
            if stats:
                info["missing_cells"] = missing
                info["target_guess"] = target
                if target and not many:
                    info["target_counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
                elif target:
                    info["target_note"] = "more than 20 distinct values (regression?)"
    except OSError as e:
        info["error"] = str(e)[:200]
    return info


def _guess_target(header: list[str]) -> str | None:
    lower = {h.strip().lower(): h for h in header}
    for name in TARGET_NAMES:
        if name in lower:
            return lower[name]
    return header[-1] if header else None


def markdown(p: dict) -> str:
    """Short Markdown summary for the chat."""
    if p.get("error"):
        return f"I could not read `{p['path']}`: {p['error']}."
    lines = [f"I found {p['n_files']} file(s) in `{p['path']}`."]
    if p["tables"]:
        lines += ["", "| File | Rows | Columns | Target guess | Classes |", "|---|---|---|---|---|"]
        for t in p["tables"]:
            classes = ""
            if t.get("target_counts"):
                total = sum(t["target_counts"].values()) or 1
                classes = ", ".join(f"{k}: {v / total:.1%}" for k, v in list(t["target_counts"].items())[:4])
            elif t.get("target_note"):
                classes = t["target_note"]
            lines.append(f"| `{t['name']}` | {t.get('rows', '?')} | {t.get('columns', '?')} | "
                         f"{'`' + t['target_guess'] + '`' if t.get('target_guess') else '–'} | {classes or '–'} |")
        missing = [t for t in p["tables"] if t.get("missing_cells")]
        lines.append("")
        lines.append("Missing values: " + (", ".join(f"`{t['name']}` {t['missing_cells']} cells" for t in missing)
                                           if missing else "none found."))
    else:
        others = ", ".join(f"`{f['name']}`" for f in p["files"][:8])
        lines.append(f"No CSV tables. Files: {others or 'none'}.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(profile(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())))
