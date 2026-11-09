"""Run shell commands where MLEvolve runs (WSL on Windows, the local shell on Linux) and map paths.

The dashboard server may run on Windows while MLEvolve runs inside WSL. Every
command the setup and run agents start goes through this module, so the rest
of the dashboard does not need to know which side it is on.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ON_WINDOWS = os.name == "nt"
_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_MNT_PATH = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")


def to_wsl_path(path: str | Path) -> str:
    """E:\\a\\b -> /mnt/e/a/b. A path that is already POSIX is returned unchanged."""
    text = str(path).strip()
    m = _WIN_PATH.match(text)
    if m:
        rest = m.group(2).replace("\\", "/").rstrip("/")
        return f"/mnt/{m.group(1).lower()}/{rest}" if rest else f"/mnt/{m.group(1).lower()}"
    return text.replace("\\", "/") if ON_WINDOWS and "\\" in text else text


def to_local_path(path: str) -> Path | None:
    """A path this process can open, or None when the path exists only inside WSL (for example /home/...)."""
    text = str(path).strip()
    if not ON_WINDOWS:
        return Path(text)
    if _WIN_PATH.match(text):
        return Path(text)
    m = _MNT_PATH.match(text)
    if m:
        return Path(f"{m.group(1).upper()}:/{m.group(2) or ''}")
    return None


def clean_path(text: str) -> str:
    """Strip the quotes and backticks people paste around a path."""
    return text.strip().strip("`'\" ").rstrip("/\\") or text.strip()


def shell_available() -> bool:
    return bool(shutil.which("wsl.exe") or shutil.which("wsl")) if ON_WINDOWS else bool(shutil.which("bash"))


def shell_argv(script: str) -> list[str]:
    if ON_WINDOWS:
        return [shutil.which("wsl.exe") or "wsl.exe", "-e", "bash", "-lc", script]
    return ["bash", "-lc", script]


def run_shell(script: str, timeout: float = 120, input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run a bash script in WSL (Windows) or locally (Linux). Never raises for a non-zero exit."""
    return subprocess.run(shell_argv(script), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, input=input_text)


# Windows: a short-lived launcher starts wsl.exe and exits at once, so the run has no living parent. Stopping
# the dashboard's process tree (a kill of the server and its children) then cannot end the run; a direct
# child died with it even with DETACHED_PROCESS. WSL ends Linux processes whose wsl.exe exits, so the
# wsl.exe process itself must live on; setsid/nohup inside WSL does not help.
_LAUNCHER = ("import subprocess, sys\n"
             "log = open(sys.argv[1], 'ab')\n"
             "p = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log, stderr=log,\n"
             "                     creationflags=0x00000200 | 0x00000008)\n"  # NEW_PROCESS_GROUP | DETACHED_PROCESS
             "print(p.pid)\n")


def spawn_detached(script_path: str, log_path: Path) -> int:
    """Start `bash <script_path>` so it keeps running when the dashboard stops. Returns the PID."""
    argv = shell_argv(f"exec bash {_sh_quote(script_path)}")
    if ON_WINDOWS:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x8)
        out = subprocess.run([sys.executable, "-c", _LAUNCHER, str(log_path), *argv], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=60, creationflags=flags)
        try:
            return int(out.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeError("could not start the run: " + (out.stderr or out.stdout)[-400:])
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return proc.pid


def pid_alive(pid) -> bool:
    """True while the process `pid` (from spawn_detached) runs. On Windows that is the wsl.exe that holds the run."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if ON_WINDOWS:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return False
            buf, size = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
            if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):  # a reused PID is not ours
                return buf.value.lower().endswith(("wsl.exe", "python.exe", "bash.exe"))
            return True
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"
