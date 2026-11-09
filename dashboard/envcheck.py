"""Environment and configuration checks for the Setup agent, and the fixes it may run.

The checks are plain commands (no LLM). The fixes are a fixed allow-list: the
chat can only start one of FIXES by its id, never an arbitrary command.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from dashboard import wslexec

REPO = Path(__file__).resolve().parent.parent
REPO_WSL = wslexec.to_wsl_path(REPO)
CORE_MODULES = ["omegaconf", "rich", "numpy", "pandas", "sklearn", "torch", "matplotlib"]
ML_MODULES = ["lightgbm", "xgboost"]
LOGIN_CACHE_S = 600
MODEL_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$")

_PROBE = r"""
cd {repo} 2>/dev/null || {{ echo repo=missing; exit 0; }}
echo repo=ok
if [ -x .venv/bin/python ]; then
  echo venv=ok
  echo pyver=$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  .venv/bin/python - <<'EOF'
import importlib.util
core = {core}
ml = {ml}
print("core_missing=" + ",".join(m for m in core if importlib.util.find_spec(m) is None))
print("ml_missing=" + ",".join(m for m in ml if importlib.util.find_spec(m) is None))
EOF
  if .venv/bin/python -c "import run" >/dev/null 2>&1; then echo import=ok; else echo import=fail; fi
else
  echo venv=missing
  command -v python3 >/dev/null && echo python3=$(python3 --version 2>&1) || echo python3=missing
fi
if command -v claude >/dev/null; then echo claude=$(claude --version 2>/dev/null | head -1); else echo claude=missing; fi
if command -v nvidia-smi >/dev/null; then echo gpus=$(nvidia-smi -L 2>/dev/null | wc -l); else echo gpus=0; fi
echo cpus=$(nproc 2>/dev/null || echo 1)
test -f .env && echo env_file=ok || echo env_file=missing
"""

# Fixes the chat may start, by id. Each is a fixed command run from the repo folder.
FIXES = {
    "create_venv": {
        "label": "Create the Python environment",
        "command": "python3 -m venv .venv",
        "timeout": 600,
    },
    "install_packages": {
        "label": "Install the MLEvolve packages",
        "command": (".venv/bin/pip install --no-deps -r requirements_base.txt && "
                    ".venv/bin/pip install --no-deps -r requirements_ml.txt && "
                    ".venv/bin/pip install --no-deps -r requirements_domain.txt"),
        "timeout": 5400,
    },
    "install_ml": {
        "label": "Install LightGBM and XGBoost",
        "command": ".venv/bin/pip install lightgbm xgboost",
        "timeout": 1800,
    },
}

_login_cache: dict = {"t": 0.0, "ok": None, "detail": ""}
_login_lock = threading.Lock()


def probe() -> dict:
    """Run the environment probe once and parse its key=value lines."""
    if not wslexec.shell_available():
        return {"shell": "missing"}
    script = _PROBE.format(repo=_q(REPO_WSL), core=json.dumps(CORE_MODULES), ml=json.dumps(ML_MODULES))
    try:
        proc = wslexec.run_shell(script, timeout=240)
    except Exception as e:  # timeout or WSL error
        return {"shell": "error", "error": str(e)[:300]}
    facts = {"shell": "ok"}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[a-z_0-9]+", key):
            facts[key] = value.strip()
    return facts


def check_login(force: bool = False) -> tuple[bool | None, str]:
    """One tiny `claude -p` call where MLEvolve runs; cached for 10 minutes."""
    with _login_lock:
        if not force and time.time() - _login_cache["t"] < LOGIN_CACHE_S and _login_cache["ok"] is not None:
            return _login_cache["ok"], _login_cache["detail"]
        script = ("cd /tmp && claude -p --model haiku --tools '' --setting-sources '' --no-session-persistence "
                  "--output-format json")
        try:
            proc = wslexec.run_shell(script, timeout=120, input_text="Reply with the single word OK.")
            data = json.loads(proc.stdout or "{}")
            ok = data.get("subtype") == "success" and not data.get("is_error")
            detail = "logged in" if ok else str(data.get("result") or proc.stderr or "no answer")[:200]
        except Exception as e:
            ok, detail = False, str(e)[:200]
        _login_cache.update(t=time.time(), ok=ok, detail=detail)
        return ok, detail


def environment_checks(facts: dict, needs_claude: bool, login: tuple[bool | None, str] | None) -> list[dict]:
    """Checklist items: {id, label, ok (True/False/None for info), detail, fix, blocking}."""
    items = []

    def add(id_, label, ok, detail="", fix=None, blocking=True):
        items.append({"id": id_, "label": label, "ok": ok, "detail": detail, "fix": fix, "blocking": blocking})

    shell = facts.get("shell")
    add("wsl", "WSL (Linux shell)" if wslexec.ON_WINDOWS else "Linux shell", shell == "ok",
        "ok" if shell == "ok" else facts.get("error", "not found"), fix="guide_wsl")
    if shell != "ok":
        return items
    add("repo", "MLEvolve folder in WSL", facts.get("repo") == "ok", REPO_WSL)
    venv = facts.get("venv") == "ok"
    add("venv", "Python environment .venv", venv,
        f"Python {facts.get('pyver')}" if venv else "missing", fix=None if venv else "create_venv")
    if venv:
        core = [m for m in facts.get("core_missing", "").split(",") if m]
        add("packages", "MLEvolve packages", not core, "all installed" if not core else "missing: " + ", ".join(core),
            fix=None if not core else "install_packages")
        if not core:
            add("import", "MLEvolve starts", facts.get("import") == "ok",
                "ok" if facts.get("import") == "ok" else "`import run` failed", fix=None if facts.get("import") == "ok" else "install_packages")
        ml = [m for m in facts.get("ml_missing", "").split(",") if m]
        add("ml", "LightGBM and XGBoost", not ml, "installed" if not ml else "missing: " + ", ".join(ml),
            fix=None if not ml else "install_ml", blocking=False)
    if needs_claude:
        cli = facts.get("claude", "missing")
        add("claude", "Claude Code CLI", cli != "missing", cli, fix=None if cli != "missing" else "guide_claude")
        if cli != "missing" and login is not None:
            add("login", "Claude Code login", login[0], login[1], fix=None if login[0] else "guide_login")
    else:
        add("env_file", "API key file .env", facts.get("env_file") == "ok", facts.get("env_file", ""), fix="guide_key")
    gpus = int(facts.get("gpus", "0") or 0)
    add("gpu", "GPU", None, f"{gpus} found" if gpus else "none, CPU only", blocking=False)
    add("cpus", "CPU cores", None, facts.get("cpus", "?"), blocking=False)
    return items


def config_checks(config: dict) -> list[dict]:
    items = []
    for key, label in (("code_model", "Code model"), ("feedback_model", "Feedback model")):
        value = str(config.get(key) or "")
        items.append({"id": key, "label": label, "ok": bool(MODEL_RE.match(value)), "detail": value or "not set",
                      "fix": None, "blocking": True})
    items.append({"id": "steps", "label": "Steps", "ok": int(config.get("steps") or 0) > 0,
                  "detail": str(config.get("steps")), "fix": None, "blocking": True})
    hours = float(config.get("time_limit_h") or 0)
    items.append({"id": "time_limit_h", "label": "Time limit", "ok": hours > 0, "detail": f"{hours:g} h",
                  "fix": None, "blocking": True})
    items.append({"id": "parallel", "label": "Parallel solutions", "ok": int(config.get("parallel") or 0) > 0,
                  "detail": str(config.get("parallel")), "fix": None, "blocking": True})
    return items


def all_ok(items: list[dict]) -> bool:
    return all(i["ok"] for i in items if i["blocking"])


def run_fix(fix_id: str) -> tuple[bool, str]:
    """Run one allow-listed fix from the repo folder. Returns (ok, last lines of output)."""
    fix = FIXES.get(fix_id)
    if fix is None:
        return False, f"unknown fix {fix_id!r}"
    try:
        proc = wslexec.run_shell(f"cd {_q(REPO_WSL)} && {fix['command']}", timeout=fix["timeout"])
    except Exception as e:
        return False, str(e)[:500]
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
    return proc.returncode == 0, tail


GUIDES = {
    "guide_wsl": ("WSL is not installed or not reachable.\n\n"
                  "1. Open **PowerShell as administrator**.\n2. Run `wsl --install -d Ubuntu`.\n"
                  "3. Restart Windows, open Ubuntu once and create your user.\n4. Press **Check again**."),
    "guide_claude": ("The Claude Code CLI is not on the PATH inside WSL.\n\n"
                     "1. In the WSL terminal, install Node.js 18 or newer.\n"
                     "2. Run `npm install -g @anthropic-ai/claude-code`.\n3. Press **Check again**."),
    "guide_login": ("Claude Code is installed but not logged in.\n\n"
                    "1. Open the WSL terminal and run `claude`.\n2. Type `/login` and follow the browser steps "
                    "with your Claude subscription.\n3. Close it with `/exit` and press **Check again**."),
    "guide_key": ("The configured models need an API key, and `.env` is missing.\n\n"
                  "Either switch both models to Claude Code (`@setup use claude code`), or run "
                  "`python run.py` once in WSL: the terminal setup asks for the key and writes `.env`."),
}


def _q(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"
