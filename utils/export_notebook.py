"""Export a finished MLEvolve run as a self-contained Jupyter notebook.

The search loop keeps generating and executing plain ``.py`` — nothing about
execution, stdout parsing or diff-based editing changes. This module only
assembles the *deliverable*: the winning solution split into cells, plus the
metrics and plots the run already wrote to ``workspace/working/``.

The notebook is never executed (a real run takes hours), so plots are embedded
from the PNGs on disk rather than re-rendered.

Usage:
    python -m utils.export_notebook                     # newest run under runs/
    python -m utils.export_notebook runs/20260802_205059_first_test
    python -m utils.export_notebook --runs-root ./runs --latest
"""

import argparse
import base64
import logging
import re
from pathlib import Path

import nbformat
import yaml

logger = logging.getLogger("MLEvolve")

# Plots are shown in this order when present; anything else follows alphabetically.
PLOT_ORDER = [
    "roc_curve.png",
    "pr_curve.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "feature_importances.png",
]

# Section headers the code agent already emits, e.g. "# ---- Step 1: Read ... ----".
SECTION_RE = re.compile(r"^#\s*-{2,}\s*(.+?)\s*-{2,}\s*$")
# Canonical jupytext cell marker, e.g. "# %%" or "# %% [markdown]".
CELL_MARKER_RE = re.compile(r"^#\s*%%")


def _md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def _code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip("\n"))


def _lift_heading(chunk: list[str]) -> tuple[str | None, str]:
    """Pull a leading ``# ---- title ----`` line out of a chunk into its heading."""
    lines = list(chunk)
    idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if idx is not None:
        section = SECTION_RE.match(lines[idx])
        if section:
            del lines[idx]
            return section.group(1), "\n".join(lines)
    return None, "\n".join(lines)


def split_into_cells(code: str) -> list[tuple[str | None, str]]:
    """Split solution source into (heading, source) chunks.

    Prefers explicit ``# %%`` markers. Falls back to the ``# ---- ... ----``
    section headers the codegen prompt asks for, then to top-level def/class
    boundaries. Returns a single chunk if none of those appear. Either way a
    section header opening a chunk becomes that cell's markdown heading, so the
    two conventions compose.
    """
    lines = code.splitlines()
    if not lines:
        return []

    use_markers = any(CELL_MARKER_RE.match(ln) for ln in lines)

    chunks: list[list[str]] = []
    current: list[str] = []
    prev_blank = True

    for line in lines:
        if use_markers:
            is_boundary = bool(CELL_MARKER_RE.match(line))
        else:
            is_boundary = bool(SECTION_RE.match(line)) or bool(
                prev_blank and re.match(r"^(def |class |@)", line)
            )

        if is_boundary and any(ln.strip() for ln in current):
            chunks.append(current)
            current = []

        # The "# %%" marker is scaffolding; section headers are kept for now and
        # lifted into headings below.
        if not (use_markers and CELL_MARKER_RE.match(line)):
            current.append(line)
        prev_blank = not line.strip()

    if any(ln.strip() for ln in current):
        chunks.append(current)

    return [_lift_heading(c) for c in chunks]


def _plot_cells(working_dir: Path) -> list[nbformat.NotebookNode]:
    """Markdown cells embedding each PNG in working/ as a base64 attachment."""
    if not working_dir.is_dir():
        return []

    pngs = sorted(working_dir.glob("*.png"))
    if not pngs:
        return []

    def sort_key(p: Path) -> tuple[int, str]:
        try:
            return (PLOT_ORDER.index(p.name), p.name)
        except ValueError:
            return (len(PLOT_ORDER), p.name)

    cells = [_md("## Figures")]
    for png in sorted(pngs, key=sort_key):
        encoded = base64.b64encode(png.read_bytes()).decode("ascii")
        title = png.stem.replace("_", " ").title()
        cell = nbformat.v4.new_markdown_cell(
            f"### {title}\n\n![{title}](attachment:{png.name})"
        )
        cell["attachments"] = {png.name: {"image/png": encoded}}
        cells.append(cell)
    return cells


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that drops unknown tags instead of failing.

    OmegaConf serialises Path fields as ``!!python/object/apply:pathlib.PosixPath``,
    which SafeLoader rejects (and which cannot be constructed on Windows anyway).
    Only plain string fields like goal/eval are needed here, so unknown tags
    become None rather than aborting the whole parse.
    """


_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)
_TolerantLoader.add_multi_constructor("!", lambda loader, suffix, node: None)


def _run_config(run_dir: Path) -> dict:
    """Load the run's saved config. Values stay unresolved, so no secrets surface."""
    raw = _read_text(run_dir / "logs" / "config.yaml")
    if not raw:
        return {}
    try:
        return yaml.load(raw, Loader=_TolerantLoader) or {}
    except yaml.YAMLError as exc:
        logger.warning(f"Could not parse saved config for {run_dir.name}: {exc}")
        return {}


def _header_cells(run_dir: Path, cfg: dict) -> list[nbformat.NotebookNode]:
    cells = [_md(f"# {run_dir.name}\n\nGenerated by MLEvolve from `logs/best_solution.py`.")]

    metric = _read_text(run_dir / "workspace" / "best_solution" / "metric.txt")
    if metric:
        cells.append(_md("## Best result\n\n```\n" + metric.strip() + "\n```"))

    goal = (cfg.get("goal") or "").strip()
    if goal:
        cells.append(_md("## Task\n\n" + goal))

    evaluation = (cfg.get("eval") or "").strip()
    if evaluation:
        cells.append(_md("## Evaluation protocol\n\n" + evaluation))

    return cells


def _metrics_cells(working_dir: Path) -> list[nbformat.NotebookNode]:
    """Prefer the markdown summary; fall back to the raw metrics table."""
    for name in ("metrics_summary.md", "metrics.md"):
        text = _read_text(working_dir / name)
        if text and text.strip():
            return [_md("## Metrics\n\n" + text.strip())]

    csv_text = _read_text(working_dir / "metrics.csv")
    if csv_text and csv_text.strip():
        return [_md("## Metrics\n\n```csv\n" + csv_text.strip() + "\n```")]
    return []


def _artifact_cell(run_dir: Path) -> nbformat.NotebookNode:
    known = [
        ("Best solution", "logs/best_solution.py"),
        ("Best submission", "workspace/best_submission/submission.csv"),
        ("Top solutions", "workspace/top_solution/"),
        ("Working files", "workspace/working/"),
        ("Search tree", "logs/journal.json"),
        ("Run log", "logs/MLEvolve.log"),
    ]
    lines = ["## Run artifacts", "", f"Root: `{run_dir}`", ""]
    lines += [
        f"- **{label}** — `{rel}`"
        for label, rel in known
        if (run_dir / rel).exists()
    ]
    return _md("\n".join(lines))


def export_notebook(run_dir: Path, out_path: Path | None = None) -> Path:
    """Build ``<run_dir>/best_solution.ipynb`` from a completed run. Returns its path."""
    run_dir = Path(run_dir).resolve()
    # A run that never found a working solution still writes a 0-byte
    # best_solution.py, so check for content rather than existence.
    candidates = [
        run_dir / "logs" / "best_solution.py",
        run_dir / "workspace" / "best_solution" / "solution.py",
        run_dir / "workspace" / "top_solution" / "top1" / "solution.py",
    ]
    solution_path = next(
        (p for p in candidates if p.exists() and p.read_text(encoding="utf-8").strip()),
        None,
    )
    if solution_path is None:
        raise FileNotFoundError(
            f"No non-empty solution found under {run_dir}; the run produced no working solution."
        )

    code = solution_path.read_text(encoding="utf-8")
    cfg = _run_config(run_dir)
    working_dir = run_dir / "workspace" / "working"

    cells = _header_cells(run_dir, cfg)
    cells += _metrics_cells(working_dir)

    cells.append(_md("## Solution"))
    for heading, chunk in split_into_cells(code):
        if heading:
            cells.append(_md(f"### {heading}"))
        cells.append(_code(chunk))

    cells += _plot_cells(working_dir)
    cells.append(_artifact_cell(run_dir))

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "mlevolve": {"run": run_dir.name, "source": str(solution_path)},
        }
    )

    out_path = Path(out_path) if out_path else run_dir / "best_solution.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out_path))
    logger.info(f"Exported notebook: {out_path}")
    return out_path


def latest_run(runs_root: Path) -> Path:
    """Most recent run directory. Names are timestamp-prefixed, so sorting works."""
    runs = sorted(p for p in Path(runs_root).glob("*") if (p / "logs").is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", help="Run directory (default: newest)")
    parser.add_argument("--runs-root", default="./runs", help="Where runs live")
    parser.add_argument("--latest", action="store_true", help="Use the newest run")
    parser.add_argument("-o", "--output", help="Output .ipynb path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.run_dir and not args.latest:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run(Path(args.runs_root))

    out = export_notebook(run_dir, Path(args.output) if args.output else None)
    print(out)


if __name__ == "__main__":
    main()
