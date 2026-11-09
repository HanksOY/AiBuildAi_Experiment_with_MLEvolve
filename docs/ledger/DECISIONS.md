# Design decisions

Choices not fixed by `PLAN.md`, newest first.

## 2026-11-08: EDA, Plan and draft review in the dashboard

**Context.** The dashboard launched runs with `eda.enabled=False`, `plan_mode.require_confirmation=False`
and `draft_review.enabled=False`, because those gates read answers from a terminal and a dashboard run has
none. So training started right after the task step, with no analysis, no plan and no approval.

**Decision.** The stages are now Environment, Configuration, Data, Task, EDA, Plan, Training, Results.
* **EDA agent and Plan agent** (`dashboard/gate_agents.py`) run before launch. Each step calls the terminal
  gates' own functions in WSL through `dashboard/gates.py` (one JSON request on stdin, one answer line):
  `generate_eda_code` + `run_eda_script`, `generate_data_plan`, `clean_task_desc` + `generate_questions`,
  `interpret_answer_input`, `interpret_review_input`, `generate_plan`, `build_augmented_task_desc` +
  `build_context_block`. Same prompts and models as the terminal. The project's settings go in as
  overrides, so questions see the real budget. The workspace is `projects/<slug>/workspace/input`, linked
  to the data. Not carried over: `detect_conflicts` and `review_settings_for_plan`; the Setup agent owns
  settings.
* Training starts only on **Approve plan and start training**. `desc_file` then holds the composed text
  (cleaned task, answers, plan, data plan). New key `plan_mode.preapproved` (default false) makes `run.py`
  use it as already clean (`task_desc_is_clean=True`), as the terminal flow does after its gates. Without it,
  `AgentSearch` would clean the text again with an LLM call that can rewrite the plan.
* **Draft review** stays inside the run. New key `draft_review.channel` (default `terminal`, unchanged):
  `file` makes `agents/draft_review.py` write `logs/draft_review.jsonl` and poll
  `logs/draft_review_answers.jsonl` (one writer per file). Answers: accept, reject with a reason, skip,
  edited, a question (answered with the existing QA prompt), or Stop from the dashboard (ends the wait).
  The wait still adds to `agent.paused_seconds`. `launch.sh` adds 2 h to its hard stop when review is on,
  because `timeout` counts wall-clock time. A new setting `draft_review` (on by default) turns it off.
* `task.require_submission=False` is passed when the data folder has no test or sample-submission file. In
  the live check, 2 correct solutions (CV AUC 0.991 and 0.996) were marked failed only because they wrote
  no `submission.csv` when there was nothing to predict.
* **Launch survives the server.** `spawn_detached` now starts `wsl.exe` through a launcher that exits at once.
  Stopping the server's process tree killed a direct child even with `DETACHED_PROCESS`; `setsid nohup`
  inside WSL does not help, because WSL ends Linux processes whose `wsl.exe` exits.
* Buttons on one-time choices (report, data plan, plan questions) close when a newer choice or a step change
  follows; a question back keeps them. Draft replies and results sort after the answer that caused them,
  because the WSL clock can be seconds behind Windows.

**Tests.** `tests/test_gate_agents.py` (14 tests, WSL calls stubbed). Live check, project `eda_plan_live`
(breast cancer, 569 rows): real EDA in 38 s, data plan changed on request, 5 real plan questions answered by
button, number and own words, a question back answered, plan approved, then launch with
`plan_mode.preapproved=True` and `draft_review.channel=file`. Run `20261108_170910_eda_plan_live`: no
second cleaning, draft 1 question answered and accepted, the server restarted during the run (the run kept
going), draft 2 rejected with "random forest only" and rewritten as a random forest, best 0.9943.

## 2026-11-07: Dashboard projects and guiding agents (MLEvolve Studio)

**Decision.** The dashboard is now organised around projects (`projects/<slug>/`, git-ignored). A centred
chat takes the user from an empty setup to a trained model with 5 agents (`dashboard/assistants.py`):
* **Setup** (environment and configuration). It runs fixed checks first (`dashboard/envcheck.py`: WSL,
  `.venv`, packages, `import run`, Claude CLI, one tiny `claude -p` login call cached for 10 min, GPU,
  settings). When all blocking checks pass, the stage is marked "ready, skipped". Otherwise it guides the
  user one item at a time. It can run only the fixes in `envcheck.FIXES`, an allow-list started by id,
  never a command from the chat. Settings change through `key=value` or plain words (one LLM call that
  returns JSON). Every change is validated against `projects.CONFIG_SPEC` and has an Undo.
* **Data** finds the folder in the message (paths may contain spaces) and profiles it with the standard
  library (`dashboard/dataprofile.py`). Files named like test, answer or submission files get row and
  column counts only, so no test labels are read.
* **Task** drafts `task.md` with the user. **Run** writes `launch.sh` (every value quoted), starts it
  detached in WSL, and links the `runs/<ts>_<slug>` folder. While the run is live, Run-agent messages
  are appended to the run's `logs/chat_requests.jsonl`, the existing path to the Main agent inside the
  run. **Analyst** answers from `review_context`.
* "Main agent (auto)" routes without an LLM call: `@name`, `/command`, a live run, a few keywords, else
  the agent of the current stage.
* Runs started outside the dashboard stay in the sidebar as read-only items with the earlier chat.
* Layout: a resizable, collapsible project sidebar; a monitor that drops down from an arrow at the top,
  can be resized by dragging, and puts the chat under it; themes Lavender (default), Bright and Dark.

**Why a hard stop in launch.sh.** Upstream does not enforce `agent.time_limit`, so `launch.sh` wraps the
run in `timeout --signal=INT` at the time limit plus 30 minutes.

**Tests.** `tests/test_projects.py` (20 tests). Live check, project `breast_cancer_live`: setup skipped
on a ready machine, plain-words settings change, Windows data path, task draft and revision, launch,
`/status` and guidance forwarded to the run, end-of-run note, Analyst answer (best 0.9990; the run
followed the guidance).

## 2026-11-06: Dashboard chat with the agents

**Decision.** The dashboard has a chat panel on every view.
* **Live run:** messages go to `logs/chat_requests.jsonl` (written only by the
  server). The run's watcher passes them, one at a time and in order, to the
  existing Main Agent (`agents/main_agent.py`), the same handler the terminal
  uses. Replies go to `logs/chat.jsonl` (written only by the run). One writer
  per file, because the run (WSL) and the server (Windows) are separate
  processes. Requests stamped before the run started (minus 60 s for clock
  drift) are ignored; requests sent during setup wait until the watcher starts.
* The Main Agent now also accepts `guide <text>` and maps free text such as
  "try XGBoost" to it. Guidance is appended to `agent.task_desc` at once as an
  "OPERATOR GUIDANCE" block, so every agent sees it in the next prompt it
  builds. Steps already running do not. It answers with a richer snapshot:
  top solutions, recent failures, guidance in effect, and what is running now
  (`monitor.current_activity()`).
* **Finished run:** the server answers with one `claude -p` call (no tools, no
  settings), from a summary of the saved run files. It cannot change the run.
  It uses the run's feedback model if that was a Claude Code model, else Sonnet.
* The server now refuses a port that is already in use. On Windows,
  SO_REUSEADDR let 2 dashboards share a port, and requests went to either one.

**Tests.** `tests/test_chat.py`. Live check `runs/20261106_193653_dashboard_smoke`:
status, guidance and a question were answered in order; every solution
started after the guidance had the block in its prompt, the first draft
(started before) did not.

## 2026-11-06: Run dashboard and its recording (`monitor.*`)

**Decision.** `python -m dashboard.server` serves `dashboard/index.html` on
127.0.0.1:8765 and reads run folders; it never imports MLEvolve. Recording is
`utils/monitor.py`, behind `monitor.enabled` (default on: it only writes
files and never changes the search):
* `llm.query` / `llm.generate` write `llm_start`/`llm_end` events to
  `logs/monitor_events.jsonl`. The agent and solution are read from the call
  stack (innermost frame under `agents/` or `engine/`, innermost local named
  `node`, `result_node`, ...), so no agent code changed. Claude Code reports
  exact usage; the streaming OpenAI/Gemini `generate` paths report none, so
  those counts are estimated at 4 characters per token and marked `estimated`.
* `run.py`'s `exec_callback` writes `exec_start`/`exec_end` and copies the
  tables and figures written to `working/` during the run into
  `workspace/node_outputs/<id>/`. Parallel solutions share `working/`, so the
  copy lists the solutions that ran at the same time (`overlapping_runs`) and
  the page warns when it is not empty. Per-solution working folders would be
  exact, but they change where generated code runs.
* Stop/Abort: the page appends to `logs/control_requests.jsonl`; a watcher
  thread forwards it to the existing `ControlBus`, the same path as the
  terminal commands. Requests need a per-server token embedded in the page and
  a localhost Origin.

**Tests.** `tests/test_monitor.py`. Live check `runs/20261106_184944_dashboard_smoke`:
19 LLM calls recorded with exact tokens, per-solution outputs copied, Stop
from the page ended the run cleanly (exit 0).

## 2026-11-04: Claude Code as provider 1 in the setup agent

**Decision.** `agents/setup_agent.py` lists "Claude Code (your Claude
subscription, no API key)" first. It asks for Claude models (`opus`, `sonnet`,
`haiku` or a full id; defaults `opus` for code, `sonnet` for feedback), checks
the login with one `claude -p` call, and asks for no key. It does not write
`.env`, so an existing key stays. `needs_setup()` returns False when both
configured models (config.yaml plus CLI overrides) are `claude-code:*`. A mixed
pair still asks for a key for the API stage.

**Tests.** `tests/test_setup_agent.py`.

## 2026-11-02: Claude Code CLI as an LLM backend (`claude-code:<model>`)

**Context.** The owner wants to run MLEvolve on their own Claude subscription
instead of a pay-per-token API key.

**Decision.** `llm/claude_code.py` runs one `claude -p` per LLM call. A model
name starting with `claude-code` routes there (`llm._provider`), for example
`agent.code.model=claude-code:opus`. Each call:
* passes `--tools ""`, so the model only answers and MLEvolve runs the code;
* passes `--setting-sources ""`, so user and project settings, hooks and
  `CLAUDE.md` stay out of the prompt (checked: with `user` the user `CLAUDE.md`
  leaked in, with `""` it did not);
* never passes `--bare`, because bare mode ignores the OAuth login;
* sends the prompt on stdin and runs in an empty temp directory;
* maps `FunctionSpec` and `json_schema` to `--json-schema` and reads
  `structured_output`.

There are no `max_tokens`, `temperature` or stop-sequence flags in the CLI, so
those arguments are ignored. The optional stage key `effort` maps to `--effort`.
Assistant prefill is appended to the message, as the GPT path already does.

**Limits.** Subscription usage limits apply. Once a limit is reached, every
call fails until it resets.

**Tests.** `tests/test_claude_code_backend.py` (CLI stubbed). Live smoke run
`runs/20261102_185209_smoke_generic_cc`: 2/2 nodes passed in generic mode.

## 2026-10-26: Switchable task framing (`task.profile`, `task.require_submission`)

**Context.** The prompts framed every task as a Kaggle competition ("Kaggle
grandmaster", "leaderboard", MLE-Bench environment notes), and the engine
failed any node that did not write `./submission/submission.csv`. We are
building a generic system for any "train a model on a dataset and report a
validation score" task.

**Decision.**
* Prompt text stays written in its Kaggle form and is passed through
  `Framing.adapt()` (`agents/prompts/framing.py`). With the defaults
  (`profile: kaggle`, `require_submission: true`), `adapt()` returns its input
  unchanged, so upstream prompts are byte-identical. `generic` applies a fixed
  rewrite table, and `require_submission: false` applies a second table that
  makes the prediction file conditional.
* `require_submission: false` also changes the engine: a node succeeds on its
  printed `Final Validation Score` alone (`result_parse_agent._determine_buggy`,
  `engine/execution.validate_executed_node`), and `save_best_solution` skips
  the copy when there is no file. `Final Validation Score` stays mandatory,
  because MCGS ranks nodes by it.
* `generic` skips the MLE-Bench format server, which only knows MLE-Bench
  competitions. A submission file that does exist still gets the content
  quality check.
* Not changed: `run_single_task.sh` and `utils/submission_fusion_utils.py`.
  Both are the MLE-Bench batch path (they assume `prepared/public` and the
  grading server). Generic runs use `python run.py task.profile=generic`.
* Not changed: the cold-start knowledge base content (built from Kaggle
  solutions). Only its label in the improve prompt is reworded.

**Why the defaults are upstream.** CLAUDE.md rule 1, and arm A0 of the
experiment protocol needs upstream behavior.

**Tests.** `pytest tests`:
* The upstream prompt snapshot (`tests/snapshots/prompts_kaggle.json`, recorded
  before any edit) must match exactly.
* An AST scan fails if a prompt string with Kaggle wording is not passed
  through `adapt()`, or if no rewrite covers it.
* Each rewrite must still match some prompt text, so a reworded prompt cannot
  silently disable its rule.

## 2026-11-09: A quiet run is not a crashed run

- Problem: the dashboard called a run "stale" after 180 s with no log write. A run that waits for a draft
  review writes nothing, so a healthy run showed "stopped responding", the chat stopped sending answers to the
  draft, and a second launch was allowed (project testing_3 got 2 runs at once).
- Fix, two signals:
  1. `agents/draft_review.py` (file channel only) touches `logs/draft_review.jsonl` every 30 s
     (`HEARTBEAT_S`) while it waits. The dashboard counts that file as run activity.
  2. The dashboard keeps the launch PID (wsl.exe on Windows) per project. A quiet run whose process
     still runs is "live" (`wslexec.pid_alive`, which also checks the image name against PID reuse).
     This also covers long LLM calls and long scripts, and runs started before the heartbeat existed.
- A live run that waits on a draft shows "waiting for your draft review" in the stage bar.
- A stale run's Analyst message and run facts now include the log end and its error lines
  (`RunReader.stop_facts`), so the Analyst reports facts and does not guess.
- Tests: `tests/test_run_liveness.py` (7 tests).

## 2026-11-09: Inspector agent and the Errors tab

Inspector agent (`dashboard/inspector.py` checks, `dashboard/inspect_agent.py` chat agent):
- Runs after each result: settings (Setup), data profile (Data), task draft (Task), EDA report and data
  plan (EDA), training plan (Plan). It runs on a background thread, so approval never waits for it.
- Two layers. Rule checks: plain code (file type from the first bytes, lock files and documents, answer or
  label files, one-value targets, metric and score line, fold-safe plan, budget estimate). Claude check: one
  `claude -p` call with the project's feedback model, at most 4 findings, told what is normal in MLEvolve
  (no test file, models chosen by the search, small data). Setup gets rule checks only.
- It warns and never blocks. Buttons: "Send N issues to the <agent>" (data notes go to the Task agent so
  they land in the task text), a one-click fix when a rule has one (the time limit), and Ignore.
- A result is identified by a digest of its content. A check that ends after the result changed is dropped.
- Setting `inspector` (default on) turns it off per project. Tests run with it off (`tests/conftest.py`).
- Budget estimate (stated in the finding): about 5 min to write a first draft, 4 min per later step, and
  about 1 min per 100 MB of data to run a script. Drafts run one at a time; later steps in parallel.
- First live test: the Claude check reported normal things (no test file, no model named). The prompt now
  lists what is normal in MLEvolve; the second run of the same draft passed with no noise.
- A settings change while the plan waits shows the plan again with the new settings (the change used to
  close the plan's Approve button).

Errors tab (`dashboard/errors.py`, `GET /api/errors?run=`):
- Reason first: running, waiting for a draft review, finished, stopped or aborted by the user, interrupted,
  hard time limit of `launch.sh`, crashed (with the traceback), killed (signal), or killed with no record.
- Then run errors, solution errors (kept apart: normal during a search), the log end and `launch.log`.
- The Analyst's end message uses the same report, with a "Show errors" button when the end was not normal.

Exit records (`utils/monitor.py`, our file): for `run.py` only, an excepthook, SIGINT/SIGTERM/SIGHUP
handlers and an atexit hook write `run_end` with reason crashed/interrupted/killed/exited. They keep the
default behavior (the signal still ends the process). Written once per run folder. Phase 1 (draft
review) had no KeyboardInterrupt handler, so Ctrl+C there left no record; this closes that gap.
A hard kill (SIGKILL, `wsl --shutdown`) still leaves no record; the report then says so.

## 2026-11-09: Retry the journal rename on the Windows drive

- **Problem:** run `20261109_152357_testing_4` finished 10 of 10 steps, then crashed in the final `save_run`. `os.replace(journal.json.tmp, journal.json)` failed with `PermissionError` on `/mnt/e`. Windows refuses to rename over a file that another program has open, and the dashboard reads `journal.json` every few seconds.
- **Decision:** `utils/serialize.py` `dump_json` tries the rename again on `PermissionError`, 10 times, 0.2 s apart, then raises. The user approved this change to an MLEvolve file. Search behaviour does not change.
- **Errors tab:** a crash in `save_run` after the log says N/N steps completed is a warning, "Crashed while saving, after all N steps finished", and lists the saved result files.
- **Not done:** opening `journal.json` in the dashboard with `FILE_SHARE_DELETE` (Windows-only code). The retry already covers the dashboard and virus scanners.
