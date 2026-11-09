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
  [cyan]guide <text>[/cyan]  tell the agents what to try or avoid in later steps
  [cyan]seed <path>[/cyan]   start a new draft from a .py file you edited
  [cyan]stop[/cyan]          finish the running steps, save, then exit
  [cyan]abort[/cyan]         kill running training now, then exit
  [cyan]help[/cyan]          this list

Anything else is read as a question or an instruction."""

# Same list without terminal markup, for the dashboard chat.
HELP_PLAIN = (
    "Commands:\n"
    "- `status`: progress, best score so far, what is running\n"
    "- `guide <text>`: tell the agents what to try or avoid in later steps\n"
    "- `seed <path>`: start a new draft from a .py file you edited\n"
    "- `stop`: finish the running steps, save, then exit\n"
    "- `abort`: kill running training now, then exit\n\n"
    "Anything else is read as a question or an instruction."
)

INTERPRET_SYSTEM = (
    "You are the operator interface for a long running machine learning search. "
    "You answer from the run snapshot, or map a request onto a command."
)

INTERPRET_TEMPLATE = """The operator typed something into a running search. Decide what it means.

Reply with JSON only:

{{
  "action": "status" | "stop" | "abort" | "seed" | "guide" | "reply",
  "path": "file path, only when action is seed",
  "guidance": "only when action is guide: the instruction for the agents, in the operator's words",
  "reply": "what to tell the operator, at most a short paragraph; Markdown allowed"
}}

Use "reply" for questions you can answer from the snapshot, and for anything you are
unsure about. Never guess a destructive action. Only choose "stop" or "abort" when the
operator clearly asks to end the run; "abort" only when they want it stopped immediately.
Choose "guide" when the operator tells the agents what to try, prefer, change or avoid in
later solutions (for example "try XGBoost" or "do not use SMOTE"). Guidance reaches the
steps that start after it is added; steps already running do not see it.

# Run snapshot
{snapshot}

# Recent conversation
{history}

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
    """Turns operator messages, typed in the terminal or sent from the dashboard, into actions.

    Replies go to the terminal and, when dashboard recording is on, to
    logs/chat.jsonl, so both places show the same conversation.
    """

    def __init__(self, cfg, bus: ControlBus, console: Console | None = None, context=None, on_guidance=None):
        self.cfg = cfg
        self.bus = bus
        self.console = console or Console()
        # Optional callable returning extra run facts (solutions, failures, guidance).
        self.context = context
        # Optional callable(text) that applies guidance at once. Without it guidance
        # is queued on the bus and applied between steps.
        self.on_guidance = on_guidance
        self._thread: threading.Thread | None = None
        self._history: list[tuple[str, str]] = []
        self._history_lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        """Start the terminal reader. Returns False when there is no terminal to read from.

        Dashboard messages do not need this: they arrive through handle_message().
        """
        if not sys.stdin.isatty():
            logger.info("Main agent terminal input disabled: stdin is not a terminal.")
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
                self.handle_message(line)

    # -- output ---------------------------------------------------------------

    def _say(self, text: str, kind: str = "reply", reply_to=None, style: str = "cyan") -> None:
        """Show a reply in the terminal and in the dashboard chat."""
        from utils import monitor

        if kind == "reply":
            self.console.print(Panel(text, border_style=style, expand=False))
        else:
            self.console.print(f"[{style}]{text}[/{style}]")
        monitor.chat_message("agent", text, kind=kind, reply_to=reply_to)
        with self._history_lock:
            self._history.append(("agent", text))
            del self._history[:-12]

    # -- input handling -------------------------------------------------------

    def handle_message(self, line: str, msg_id=None) -> None:
        """Entry point for one operator message. Never raises.

        msg_id is set for dashboard messages, which the dashboard already shows;
        terminal input is copied into the chat log so both places match.
        """
        if msg_id is None:
            from utils import monitor

            monitor.chat_message("operator", line, kind="terminal")
        with self._history_lock:
            self._history.append(("operator", line))
            del self._history[:-12]
        try:
            self._handle(line, msg_id)
        except Exception as e:
            logger.warning(f"Main agent could not handle input {line!r}: {e}")
            self._say(f"Could not handle that: {e}", kind="error", reply_to=msg_id, style="red")

    def _handle(self, line: str, msg_id=None) -> None:
        word, _, rest = line.partition(" ")
        command, rest = word.lower().lstrip("/"), rest.strip()

        if command in {"help", "?", "h"}:
            self._say(HELP_PLAIN, reply_to=msg_id)
            return
        if command in {"status", "st"}:
            self._say(self.status_text(), reply_to=msg_id)
            return
        if command in {"stop", "quit", "exit"}:
            self._request("stop", "Stopping after the running steps finish.", msg_id)
            return
        if command in {"abort", "kill"}:
            self._request("abort", "Aborting: killing running training now.", msg_id)
            return
        if command == "seed":
            self._queue_seed(rest, msg_id)
            return
        if command == "guide":
            self._queue_guidance(rest, msg_id)
            return

        self._interpret(line, msg_id)

    def _request(self, action: str, text: str, msg_id=None) -> None:
        if action == "abort":
            self.bus.request_abort()
        else:
            self.bus.request_stop()
        self.bus.push(action)
        self._say(text, kind="action", reply_to=msg_id, style="red" if action == "abort" else "yellow")

    def _queue_seed(self, raw: str, msg_id=None) -> None:
        if not raw:
            self._say("Usage: `seed <path to a .py file>`", kind="error", reply_to=msg_id, style="yellow")
            return
        path = Path(raw).expanduser()
        if not path.is_file():
            self._say(f"Not a file: {path}", kind="error", reply_to=msg_id, style="red")
            return
        self.bus.push("seed", path=str(path.resolve()))
        self._say(f"Queued: the next step will start from {path.name}.", kind="action", reply_to=msg_id, style="green")

    def _queue_guidance(self, text: str, msg_id=None, note: str = "") -> None:
        text = text.strip()
        if not text:
            self._say("Usage: `guide <what the agents should try or avoid>`", kind="error",
                      reply_to=msg_id, style="yellow")
            return
        text = text[:2000]
        if self.on_guidance is not None:
            self.on_guidance(text)
        else:
            self.bus.push("guide", text=text)
        self._say((note + "\n\n" if note else "") +
                  f"Guidance added for the steps that start from now on:\n\n> {text}",
                  kind="action", reply_to=msg_id, style="green")

    def status_text(self) -> str:
        snap = self._snapshot()
        rows = [("Progress", "progress"), ("Running", "running"), ("Best score", "best_metric"),
                ("Best solution", "best_node"), ("Elapsed", "elapsed"), ("Solutions", "nodes"),
                ("Branches", "branches")]
        lines = [f"- **{label}:** {snap[key]}" for label, key in rows if snap.get(key) is not None]
        if not lines:
            # The loop publishes progress only once the search proper starts.
            lines.append(f"- **Phase:** setting up and writing the first drafts "
                         f"({snap.get('solutions_so_far', 0)} solutions so far)")
        for item in snap.get("in_progress") or []:
            if item.get("what") == "solution_script":
                lines.append(f"- **Now:** running the script of solution {item['solution']} ({item['seconds']} s)")
            else:
                target = f"solution {item['solution']}" if item.get("solution") else "a new solution"
                lines.append(f"- **Now:** {item.get('agent') or 'an agent'} ({item.get('step')}) for {target} "
                             f"({item['seconds']} s)")
        for g in snap.get("operator_guidance_in_effect") or []:
            lines.append(f"- **Guidance in effect:** {g}")
        return "\n".join(lines)

    def _snapshot(self) -> dict:
        snap = dict(self.bus.snapshot)
        if self.context is not None:
            try:
                snap.update(self.context())
            except Exception as e:
                logger.warning(f"Main agent context failed: {e}")
        return snap

    def _interpret(self, message: str, msg_id=None) -> None:
        """Free text. Interpreted here so the search loop never waits on a model."""
        from llm import generate
        from utils.response import extract_review

        self.console.print("[dim]thinking ...[/dim]")
        with self._history_lock:
            history = "\n".join(f"{who}: {text[:600]}" for who, text in self._history[:-1]) or "(none)"
        try:
            raw = generate(
                prompt={
                    "system": INTERPRET_SYSTEM,
                    "user": INTERPRET_TEMPLATE.format(
                        snapshot=json.dumps(self._snapshot(), indent=2, default=str),
                        history=history, message=message,
                    ),
                },
                cfg=self.cfg,
                stage="feedback",
            )
            data = extract_review(raw)
        except Exception as e:
            logger.warning(f"Main agent interpretation failed: {e}")
            self._say("I did not follow that. Type `help` for commands.", kind="error",
                      reply_to=msg_id, style="yellow")
            return

        action = str(data.get("action", "reply")).lower() if isinstance(data, dict) else "reply"
        reply = str(data.get("reply", "")).strip() if isinstance(data, dict) else ""

        if action == "stop":
            self._request("stop", reply or "Stopping after the running steps finish.", msg_id)
        elif action == "abort":
            self._request("abort", reply or "Aborting now.", msg_id)
        elif action == "seed":
            self._queue_seed(str(data.get("path", "")), msg_id)
        elif action == "guide":
            self._queue_guidance(str(data.get("guidance") or message), msg_id, note=reply)
        elif action == "status":
            self._say((reply + "\n\n" if reply else "") + self.status_text(), reply_to=msg_id)
        else:
            self._say(reply or "I am not sure what you mean. Type `help` for commands.", reply_to=msg_id)


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
