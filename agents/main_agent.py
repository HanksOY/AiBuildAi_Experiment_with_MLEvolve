"""Main agent: a command channel that stays open while the search runs.

Until now every input path closed before the search started. This adds one that
stays open, without restructuring the thread pool.

The design is deliberately cooperative rather than preemptive. Sub agents are
plain function calls on worker threads, so nothing can be messaged mid flight,
and an LLM request already in the air cannot be recalled. So commands are
queued by a reader thread and drained by the search loop between steps, at
points where stopping leaves the journal consistent.

Two stop tiers, because they answer different needs:
  stop   set a flag; running steps finish, results are saved, the loop exits.
  abort  same, plus running training subprocesses are killed immediately.

Typing anything that is not a command routes to the feedback model, which
either answers from the live run snapshot or maps the request onto a command.
That interpretation happens on the reader thread so the search never blocks on
it.
"""

import json
import logging
import queue
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger("MLEvolve")

HELP_TEXT = """[bold]Commands[/bold]
  [cyan]status[/cyan]        progress, best score so far, what is running
  [cyan]seed <path>[/cyan]   start a new draft from a .py file you edited
  [cyan]stop[/cyan]          finish the running steps, save, then exit
  [cyan]abort[/cyan]         kill running training now, then exit
  [cyan]help[/cyan]          this list

Anything else is read as a question or an instruction."""

INTERPRET_SYSTEM = (
    "You are the operator interface for a long running machine learning search. "
    "You answer briefly from the run snapshot, or map a request onto a command."
)

INTERPRET_TEMPLATE = """The operator typed something into a running search. Decide what it means.

Reply with JSON only:

{{
  "action": "status" | "stop" | "abort" | "seed" | "reply",
  "path": "file path, only when action is seed",
  "reply": "what to tell the operator, one or two sentences"
}}

Use "reply" for questions you can answer from the snapshot, and for anything you are
unsure about — never guess a destructive action. Only choose "stop" or "abort" when the
operator clearly asks to end the run; "abort" only when they want it stopped immediately.

# Run snapshot
{snapshot}

# What the operator typed
{message}
"""


class ControlBus:
    """Thread safe channel between the reader thread and the search loop."""

    def __init__(self):
        self._actions: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._abort = threading.Event()
        self._snapshot: dict = {}
        self._lock = threading.Lock()

    # -- producer side, called from the reader thread -------------------------

    def push(self, action: str, **payload) -> None:
        self._actions.put((action, payload))

    def request_stop(self) -> None:
        self._stop.set()

    def request_abort(self) -> None:
        self._abort.set()
        self._stop.set()

    # -- consumer side, called from the search loop ---------------------------

    def drain(self) -> list[tuple[str, dict]]:
        items = []
        while True:
            try:
                items.append(self._actions.get_nowait())
            except queue.Empty:
                return items

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def abort_requested(self) -> bool:
        return self._abort.is_set()

    # -- shared state ---------------------------------------------------------

    def publish(self, **fields) -> None:
        with self._lock:
            self._snapshot.update(fields)

    @property
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)


class MainAgent:
    """Reads operator input on a background thread and queues actions for the loop."""

    def __init__(self, cfg, bus: ControlBus, console: Console | None = None):
        self.cfg = cfg
        self.bus = bus
        self.console = console or Console()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        """Start the reader. Returns False when there is no terminal to read from."""
        if not sys.stdin.isatty():
            logger.info("Main agent disabled: stdin is not a terminal.")
            return False

        self.console.print(
            Panel(
                HELP_TEXT,
                title="[bold]Main agent is listening[/bold]",
                subtitle="[dim]type at any time; commands apply between steps[/dim]",
                border_style="green",
                expand=False,
            )
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="main-agent")
        self._thread.start()
        return True

    def _read_loop(self) -> None:
        while not self.bus.stop_requested:
            try:
                line = sys.stdin.readline()
            except (ValueError, OSError):
                return
            if not line:  # EOF
                return
            line = line.strip()
            if line:
                try:
                    self._handle(line)
                except Exception as e:
                    logger.warning(f"Main agent could not handle input {line!r}: {e}")
                    self.console.print(f"[red]Could not handle that: {e}[/red]")

    # -- input handling -------------------------------------------------------

    def _handle(self, line: str) -> None:
        word, _, rest = line.partition(" ")
        command, rest = word.lower(), rest.strip()

        if command in {"help", "?", "h"}:
            self.console.print(Panel(HELP_TEXT, border_style="cyan", expand=False))
            return
        if command in {"status", "st"}:
            self.bus.push("status")
            return
        if command in {"stop", "quit", "exit"}:
            self.console.print("[yellow]Stopping after the running steps finish ...[/yellow]")
            self.bus.request_stop()
            self.bus.push("stop")
            return
        if command in {"abort", "kill"}:
            self.console.print("[red]Aborting: killing running training now ...[/red]")
            self.bus.request_abort()
            self.bus.push("abort")
            return
        if command == "seed":
            self._queue_seed(rest)
            return

        self._interpret(line)

    def _queue_seed(self, raw: str) -> None:
        if not raw:
            self.console.print("[yellow]Usage: seed <path to a .py file>[/yellow]")
            return
        path = Path(raw).expanduser()
        if not path.is_file():
            self.console.print(f"[red]Not a file:[/red] {path}")
            return
        self.console.print(f"[green]Queued:[/green] next step will start from {path.name}")
        self.bus.push("seed", path=str(path.resolve()))

    def _interpret(self, message: str) -> None:
        """Free text. Interpreted here so the search loop never waits on a model."""
        from llm import generate
        from utils.response import extract_review

        snapshot = self.bus.snapshot
        self.console.print("[dim]thinking ...[/dim]")
        try:
            raw = generate(
                prompt={
                    "system": INTERPRET_SYSTEM,
                    "user": INTERPRET_TEMPLATE.format(
                        snapshot=json.dumps(snapshot, indent=2, default=str), message=message
                    ),
                },
                cfg=self.cfg,
                stage="feedback",
            )
            data = extract_review(raw)
        except Exception as e:
            logger.warning(f"Main agent interpretation failed: {e}")
            self.console.print("[yellow]I did not follow that. Type 'help' for commands.[/yellow]")
            return

        action = str(data.get("action", "reply")).lower() if isinstance(data, dict) else "reply"
        reply = str(data.get("reply", "")).strip() if isinstance(data, dict) else ""

        if action == "stop":
            self.console.print(f"[yellow]{reply or 'Stopping after the running steps finish.'}[/yellow]")
            self.bus.request_stop()
            self.bus.push("stop")
        elif action == "abort":
            self.console.print(f"[red]{reply or 'Aborting now.'}[/red]")
            self.bus.request_abort()
            self.bus.push("abort")
        elif action == "seed":
            self._queue_seed(str(data.get("path", "")))
        elif action == "status":
            self.bus.push("status")
        else:
            self.console.print(Panel(reply or "I am not sure what you mean.",
                                     border_style="cyan", expand=False))


def _demo() -> None:
    """Interactive demo: the real agent driving a fake search loop.

    Mirrors run.py's control flow with 3 second steps and no LLM calls for the
    work itself, so the command channel can be exercised in seconds instead of
    hours. Free text still uses the configured feedback model when available.
    """
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    console = Console()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import _load_cfg

        cfg = _load_cfg(use_cli_args=False)
        console.print("[dim]Config loaded — free text will use the feedback model.[/dim]")
    except Exception as e:
        cfg = None
        console.print(f"[yellow]No config ({e}); only exact commands will work.[/yellow]")

    total_steps, workers, started = 12, 3, time.time()
    bus = ControlBus()
    if not MainAgent(cfg, bus, console).start():
        console.print("[red]No terminal detected — run this from an interactive shell.[/red]")
        return

    console.print(f"[green]Simulating a {total_steps} step search (3s per step).[/green]\n")

    best = [None]
    seeds: list[str] = []

    def fake_step(label):
        time.sleep(3)
        score = round(random.uniform(0.02, 0.18), 4)
        if best[0] is None or score > best[0]:
            best[0] = score
        return label, score

    completed, submitted = 0, 0
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = set()
    for _ in range(workers):
        futures.add(executor.submit(fake_step, f"step{submitted}")); submitted += 1

    try:
        while completed < total_steps:
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

            for action, payload in bus.drain():
                if action == "status":
                    render_status(bus, console)
                elif action == "seed":
                    seeds.append(payload["path"])
                    console.print(f"[green]Seed queued:[/green] {Path(payload['path']).name}")
                elif action == "abort":
                    console.print("[red](would call interpreter.terminate_all_subprocesses())[/red]")

            bus.publish(
                progress=f"{completed}/{total_steps} steps",
                running=len(futures),
                best_metric=best[0],
                elapsed=f"{(time.time() - started) / 60:.1f} min",
                nodes=completed + 1,
                branches=min(3, completed + 1),
            )

            if bus.abort_requested:
                console.print(f"[red]Aborted at {completed}/{total_steps}.[/red]")
                break
            if bus.stop_requested and not futures:
                console.print(f"[yellow]Stopped cleanly at {completed}/{total_steps}.[/yellow]")
                break
            if not done:
                continue

            for fut in done:
                futures.remove(fut)
                label, score = fut.result()
                completed += 1
                console.print(f"  [dim]{label} finished: {score}[/dim]")

                if bus.stop_requested:
                    console.print("  [dim](stop requested — submitting nothing further)[/dim]")
                elif seeds:
                    seed = seeds.pop(0)
                    console.print(f"  [green]drafting from {Path(seed).name}[/green]")
                    futures.add(executor.submit(fake_step, f"seed:{Path(seed).name}"))
                elif completed + len(futures) < total_steps:
                    futures.add(executor.submit(fake_step, f"step{submitted}")); submitted += 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Ctrl+C[/yellow]")
    finally:
        executor.shutdown(wait=not bus.abort_requested, cancel_futures=bus.abort_requested)

    console.print(
        f"\n[bold]Done.[/bold] completed={completed}/{total_steps}  best={best[0]}  "
        f"stop={bus.stop_requested}  abort={bus.abort_requested}"
    )


def render_status(bus: ControlBus, console: Console) -> None:
    """Print the live snapshot the loop keeps up to date."""
    snap = bus.snapshot
    if not snap:
        console.print("[dim]No progress recorded yet.[/dim]")
        return

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    for label, key in [
        ("Progress", "progress"),
        ("Running", "running"),
        ("Best score", "best_metric"),
        ("Best node", "best_node"),
        ("Elapsed", "elapsed"),
        ("Nodes", "nodes"),
        ("Branches", "branches"),
    ]:
        if snap.get(key) is not None:
            table.add_row(label, str(snap[key]))
    console.print(Panel(table, title="[bold]Run status[/bold]", border_style="cyan", expand=False))


if __name__ == "__main__":
    _demo()
