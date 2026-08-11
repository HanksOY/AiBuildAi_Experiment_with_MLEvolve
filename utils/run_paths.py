"""Where a run's final outputs will land, reported up front.

Every path below is fixed once ``prep_cfg`` stamps ``exp_name``, so the full
list can be shown at startup even though the files themselves appear later.
"""

import logging
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger("MLEvolve")

_WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")

# (label, path relative to the run root, appears only after the run finishes)
FINAL_OUTPUTS = [
    ("Notebook", "best_solution.ipynb"),
    ("Best solution", "logs/best_solution.py"),
    ("Best submission", "workspace/best_submission/submission.csv"),
    ("Top solutions", "workspace/top_solution/"),
    ("Plots & metrics", "workspace/working/"),
    ("Search tree", "logs/journal.json"),
    ("Run log", "logs/MLEvolve.log"),
]


def to_windows_path(path) -> str:
    """Render a WSL path in Windows form so it can be opened from the host.

    ``/mnt/e/foo`` becomes ``E:\\foo``. Paths outside /mnt (the WSL rootfs) and
    paths that are already native are returned unchanged.
    """
    # Normalise separators first: a POSIX path that has been through pathlib on a
    # Windows host arrives as "\mnt\e\..." and would otherwise miss the match.
    text = str(path).replace("\\", "/")
    match = _WSL_MOUNT_RE.match(text)
    if not match:
        return str(path)
    drive, rest = match.groups()
    return f"{drive.upper()}:\\" + rest.replace("/", "\\")


def run_root(cfg) -> Path:
    """Run directory holding both logs/ and workspace/."""
    log_dir = Path(cfg.log_dir)
    workspace_dir = Path(cfg.workspace_dir)
    if log_dir.parent == workspace_dir.parent:
        return log_dir.parent
    return log_dir


def output_paths(cfg) -> list[tuple[str, str]]:
    """(label, absolute path) for each final output, in display order.

    Joined as text rather than via pathlib so a WSL path stays POSIX-shaped
    until to_windows_path converts it.
    """
    root = str(run_root(cfg)).replace("\\", "/").rstrip("/")
    return [
        (label, to_windows_path(f"{root}/{rel}".rstrip("/")))
        for label, rel in FINAL_OUTPUTS
    ]


def print_output_paths(cfg, console: Console | None = None) -> None:
    """Show the run's output locations before the search loop starts."""
    console = console or Console()

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white", overflow="fold")
    for label, path in output_paths(cfg):
        table.add_row(label, path)

    console.print(
        Panel(
            table,
            title="[bold]Outputs for this run[/bold]",
            subtitle="[dim]created as the run progresses[/dim]",
            border_style="green",
            expand=False,
        )
    )

    logger.info(f"Run outputs will be written under: {run_root(cfg)}")
    for label, path in output_paths(cfg):
        logger.info(f"  {label}: {path}")
