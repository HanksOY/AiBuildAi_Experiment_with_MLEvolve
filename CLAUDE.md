# CLAUDE.md

## What this repo is

A fork of MLEvolve (github.com/InternScience/MLEvolve), an MLE agent that solves Kaggle style tasks with Monte Carlo Graph Search (MCGS), multi agent planning and coding, and a global memory retrieved with BM25 plus FAISS.

We are adding an **Experience Ledger**: a persistent library of reusable ideas ("tips") with outcome records. It ranks which tips the planner is offered, using relevance and track record, and carries evidence across tasks.

The full build plan is in `PLAN.md`. Work through it one phase at a time, in order.

## Ground rules

1. **Preserve upstream behavior.** Every new behavior sits behind a `ledger.*` config key. With `ledger.enabled: false`, MLEvolve must behave exactly as upstream. Keep a regression test for this and run it in every phase.
2. **Do not change MCGS.** UCT, exploration decay, the explore and exploit switch, stagnation detection, fusion, rewards, and backpropagation stay as they are. Add hook calls only.
3. **No new LLM calls.** The ledger adds no LLM calls of its own unless `PLAN.md` says so explicitly.
4. **No test data.** The ledger uses only signals MLEvolve already uses: validation metrics and predictions from the agent's own split of the training data. Never read, store, or log test labels, grader output, or leaderboard results.
5. **Frozen evaluation.** In `ledger.mode: eval_frozen`, the persistent store is read only. Task local updates go to an overlay that is discarded at task end.
6. **Pure math, tested.** All algorithms live in pure functions under `ledger/core/`, with no LLM, GPU, network, or file access. Write unit tests first, using the worked examples in `PLAN.md` section 5 as test cases.
7. **Reproducible.** Seed every random draw (Thompson draws, bootstrap resamples) from a per task seed, and log the seed.
8. **Tune only on training tasks.** Never adjust defaults based on evaluation task results.
9. **Small steps.** One phase per branch or PR, small commits, and an entry in `docs/ledger/DECISIONS.md` for any design choice not fixed by `PLAN.md`.

## Stop and ask before

* Editing any MLEvolve file beyond adding a hook call or a flagged prompt section.
* Changing a default value from `PLAN.md` section 5.11.
* Reading any data outside the task's training data.
* Starting GPU runs longer than a smoke test.

## Where things are (confirm and complete in Phase 0)

* `run.py` and `run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]` start a task.
* `config/config.yaml` holds settings, including `agent.use_global_memory`, `agent.memory_embedding_model_path`, `agent.steps`, and `agent.time_limit`.
* `agents/`, `engine/`, `llm/`, and `utils/` hold the agents, search engine, LLM clients, and helpers.
* Runs write to `./runs/<timestamp>_<exp_id>/`: search tree logs, best solution code, top K submissions.
* New code goes in `ledger/`. Ledger docs go in `docs/ledger/`, including `MLEVOLVE_MAP.md` from Phase 0.

## Commands (fill in during Phase 1)

* Unit tests: `pytest ledger/tests`
* Regression test: to be defined in Phase 1
* Smoke task: to be defined in Phase 1 (small task, short `agent.time_limit`, LLM stub where possible)

## A phase is done when

* All acceptance criteria in `PLAN.md` for that phase pass.
* Unit tests and the upstream regression test pass.
* `docs/ledger/DECISIONS.md` and `docs/ledger/MLEVOLVE_MAP.md` are up to date.
