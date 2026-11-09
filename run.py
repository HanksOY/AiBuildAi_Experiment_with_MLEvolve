import atexit
import logging
import sys
import shutil
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from utils.logging_config import setup_logging
from utils.run_paths import print_output_paths, run_root, to_windows_path
from utils.export_notebook import export_notebook
from utils.data_preview import clean_task_desc
from utils import monitor
from agents.prompts.framing import framing_of
from agents.plan_mode import run_plan_gate
from agents.setup_agent import needs_setup, run_setup
from agents.eda_agent import run_eda_gate, build_context_block
from agents.main_agent import ControlBus, MainAgent, render_status
from agents.draft_review import review_draft, ACCEPT, REJECT, ABORT
from rich.console import Console
import torch



def run():
    # First run: no key means nothing below can work, so set one up before
    # load_cfg() fails on a half-filled config.
    if needs_setup():
        if not run_setup():
            return
    cfg = load_cfg()
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    monitor.run_started(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    print_output_paths(cfg)

    # EDA gate. Runs first so the plan that follows is grounded in measured
    # facts rather than assumptions, and so the confirmed data plan reaches
    # every agent through the task description.
    eda_block = ""
    if cfg.eda.enabled:
        from utils.data_preview import generate as preview_data

        try:
            preview = preview_data(str(cfg.workspace_dir), framing=framing_of(cfg))
        except Exception as e:
            logger.warning(f"Data preview failed before EDA: {e}")
            preview = ""
        result = run_eda_gate(cfg, task_desc, preview)
        if result is None and sys.stdin.isatty() and cfg.eda.enabled:
            logger.info("No EDA report; continuing without one.")
        elif result:
            report_path, data_plan = result
            eda_block = build_context_block(report_path, data_plan)

    # Plan gate. Runs after the workspace exists (clean_task_desc reads
    # input/ for a sample submission) and before anything is spent on search.
    cleaned_task_desc = task_desc
    task_desc_is_clean = False
    if cfg.plan_mode.require_confirmation:
        cleaned_task_desc = clean_task_desc(task_desc, cfg)
        # The gate returns the task description augmented with the user's
        # answers and the approved plan, so what they agreed to is what the
        # agents actually receive.
        approved_task_desc = run_plan_gate(cfg, task_desc, cleaned_task_desc)
        if approved_task_desc is None:
            shutil.rmtree(cfg.workspace_dir, ignore_errors=True)
            logger.info("Exiting at the plan gate; nothing was executed.")
            return
        cleaned_task_desc = approved_task_desc
        task_desc_is_clean = True

    # The confirmed data plan rides along with the task description, so the
    # data_processing_and_feature_engineering step works from it.
    if eda_block:
        from agents.plan_mode import _as_text

        cleaned_task_desc = _as_text(cleaned_task_desc) + eda_block
        task_desc_is_clean = True

    # Started from the dashboard: the EDA and plan steps already ran there, and
    # desc_file holds the approved, already cleaned task description.
    if getattr(cfg.plan_mode, "preapproved", False):
        task_desc_is_clean = True

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=cleaned_task_desc,
        cfg=cfg,
        journal=journal,
        task_desc_is_clean=task_desc_is_clean,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        node_id = args[1] if len(args) > 1 else kwargs.get("id")
        t0 = monitor.exec_started(node_id)
        ok = False
        try:
            res = interpreter.run(*args, **kwargs)
            ok = True
        finally:
            monitor.exec_finished(node_id, t0, ok=ok)
        status.update("[green]Generating code...")
        return res

    def export_final_notebook():
        """Export the winning solution as a notebook. Never fatal to the run."""
        try:
            path = export_notebook(run_root(cfg))
            logger.info(f"📓 Notebook exported: {to_windows_path(path)}")
        except FileNotFoundError as e:
            logger.info(f"Skipping notebook export: {e}")
        except Exception as e:
            logger.warning(f"Notebook export failed: {e}")

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    def step_task_from_seed(path):
        """Start a fresh draft from a file the operator edited."""
        logger.info(f"[step_task_from_seed] Drafting from {path}")
        return agent.step(exec_callback=exec_callback, node=None, init_solution_path=path)

    console = Console()
    bus = ControlBus()
    operator_guidance: list[str] = []

    def run_context() -> dict:
        """Facts the main agent may use when it answers the operator."""
        nodes = [n for n in list(journal.nodes) if n.stage != "root"]
        scored = [n for n in nodes if not n.is_buggy and n.metric is not None and n.metric.value is not None]
        scored.sort(key=lambda n: n.metric.value, reverse=bool(agent.metric_maximize))
        plan_head = lambda n: " ".join((n.plan or "").split())[:240]
        activity = monitor.current_activity()
        known = {n.id[:8] for n in nodes}
        for item in activity:  # a draft's calls name the hidden search root
            if item.get("solution") not in known:
                item["solution"] = None
        return {
            "metric_higher_is_better": agent.metric_maximize,
            "top_solutions": [{"id": n.id[:8], "stage": n.stage, "score": n.metric.value, "plan": plan_head(n)}
                              for n in scored[:5]],
            "recent_failures": [{"id": n.id[:8], "stage": n.stage, "error": n.exc_type, "plan": plan_head(n)}
                                for n in [n for n in nodes if n.is_buggy][-3:]],
            "operator_guidance_in_effect": operator_guidance,
            "solutions_so_far": len(nodes),
            "in_progress": activity,
            "time_limit_s": cfg.agent.time_limit,
        }

    def apply_guidance(text: str) -> None:
        """Operator guidance from the main agent: appended to the task description every agent reads."""
        from agents.plan_mode import _as_text

        operator_guidance.append(text)
        stamp = time.strftime("%H:%M")
        agent.task_desc = (
            _as_text(agent.task_desc)
            + f"\n\n**OPERATOR GUIDANCE (added during the run at {stamp})**\n"
            + "The task owner added this while the search was running. Follow it in new solutions "
            + "unless it conflicts with the task's metric or data rules; say so if you deviate.\n"
            + f"- {text}\n"
        )
        logger.info(f"Operator guidance added: {text}")

    # Guidance is applied at once (not between steps), so it also reaches the first drafts.
    main_agent = MainAgent(cfg, bus, console, context=run_context, on_guidance=apply_guidance)
    main_agent.start()
    monitor.start_control_watcher(cfg, bus, chat_handler=main_agent.handle_message)
    agent.control = bus  # lets a queued stop land inside a step, not just between them

    def publish_snapshot(completed, total, running):
        best = agent.best_node
        bus.publish(
            progress=f"{completed}/{total} steps",
            running=running,
            best_metric=best.metric.value if (best and best.metric) else None,
            best_node=best.id if best else None,
            elapsed=f"{agent.elapsed() / 60:.0f} min",
            nodes=len(journal),
            branches=len(agent.branch_all_nodes),
        )

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    lock = threading.Lock()
    completed = 0

    pending_draft_nodes = []
    if initial_draft_count > 0 and total_steps > 0:
        logger.info(f"📝 Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only():
            logger.info(f"[step_task_generate_only] Generating draft from virtual root")
            return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)

        def discard_draft(node):
            """Unhook a rejected draft so the tree does not keep a child that never runs."""
            try:
                agent.virtual_root.children.discard(node)
            except AttributeError:
                pass
            agent.virtual_root.sub_expected_child_count()
            branch = agent.branch_all_nodes.get(getattr(node, "branch_id", None))
            if branch and node in branch:
                branch.remove(node)

        target_drafts = min(initial_draft_count, total_steps)
        max_regen = getattr(cfg.draft_review, "max_regenerations", 2)

        for draft_idx in range(target_drafts):
            # Drafts are generated sequentially, so this is a clean place to stop.
            if bus.stop_requested:
                logger.info("Stop requested during draft generation; moving on with what exists.")
                break

            attempts = 0
            while True:
                try:
                    logger.info(f"Generating draft {draft_idx + 1}/{target_drafts} (code only)")
                    cur_node = step_task_generate_only()
                except Exception as e:
                    logger.exception(f"Exception during draft {draft_idx + 1} generation: {e}")
                    break

                verdict = review_draft(agent, cur_node, draft_idx + 1, target_drafts, console)

                if verdict == ACCEPT:
                    pending_draft_nodes.append(cur_node)
                    logger.info(f"Draft {draft_idx + 1} accepted: node.id={cur_node.id}")
                    break

                if verdict == ABORT:
                    logger.info("User aborted at the draft review; stopping the run.")
                    bus.request_stop()
                    discard_draft(cur_node)
                    break

                # Rejected. Carry any feedback into the replacement.
                discard_draft(cur_node)
                agent.draft_feedback = getattr(cur_node, "review_feedback", None)
                attempts += 1
                if attempts > max_regen:
                    console.print(
                        f"[yellow]Draft {draft_idx + 1} rejected {attempts} times; "
                        f"moving on without it.[/yellow]"
                    )
                    logger.info(f"Draft {draft_idx + 1} abandoned after {attempts} rejections.")
                    break
                console.print(f"[dim]Regenerating draft {draft_idx + 1} ...[/dim]")

            agent.draft_feedback = None
            if bus.stop_requested:
                break

        logger.info(f"✅ Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

    if pending_draft_nodes or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = set()
            for i, node in enumerate(pending_draft_nodes):
                futures.add(executor.submit(execute_draft_node, node))
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            initial_step_tasks = min(max_workers, total_steps - completed) - len(pending_draft_nodes)
            if initial_step_tasks > 0:
                for _ in range(initial_step_tasks):
                    futures.add(executor.submit(step_task))
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            pending_seeds = []

            def pump_commands():
                """Drain queued operator commands. Runs on the loop thread, between steps."""
                for action, payload in bus.drain():
                    if action == "status":
                        publish_snapshot(completed, total_steps, len(futures))
                        render_status(bus, console)
                    elif action == "guide":
                        apply_guidance(payload["text"])
                    elif action == "seed":
                        pending_seeds.append(payload["path"])
                        logger.info(f"Operator queued a seed solution: {payload['path']}")
                    elif action == "abort":
                        logger.info("Operator requested abort; terminating subprocesses.")
                        interpreter.terminate_all_subprocesses()
                    elif action == "stop":
                        logger.info("Operator requested stop; no further steps will be submitted.")

            while completed < total_steps:
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                # The 1s tick is the natural drain point: frequent enough to feel
                # responsive, and always between step boundaries.
                pump_commands()
                publish_snapshot(completed, total_steps, len(futures))

                if bus.abort_requested:
                    logger.info("Aborting: cancelling outstanding work.")
                    console.print("[red]Aborted. Saving what completed ...[/red]")
                    break
                if bus.stop_requested and not futures:
                    logger.info("Stop requested and nothing is running; exiting the loop.")
                    console.print("[yellow]Stopped.[/yellow]")
                    break

                if not done:
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                for fut in done:
                    futures.remove(fut)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1  # Exclude virtual node
                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                    # A queued stop means no new work, but whatever is already
                    # running is allowed to finish and be saved.
                    if bus.stop_requested:
                        logger.info("Stop requested; not submitting further steps.")
                    elif pending_seeds:
                        seed = pending_seeds.pop(0)
                        futures.add(executor.submit(step_task_from_seed, seed))
                        logger.info(f"📤 Submitted operator seed: {seed}")
                        console.print(f"[green]Started a draft from {Path(seed).name}[/green]")
                    elif completed + len(futures) < total_steps:
                        futures.add(executor.submit(step_task, cur_node))
                        logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            monitor.run_ended("interrupted")
            interpreter.terminate_all_subprocesses()
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            export_final_notebook()
            raise
        finally:
            if interrupted:
                pass
            elif bus.abort_requested:
                # Training is already dead; drop anything still queued.
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
            with lock:
                save_run(cfg, journal)
    else:
        logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)
    monitor.run_ended("aborted" if bus.abort_requested else "stopped" if bus.stop_requested else "finished")
    export_final_notebook()
    print_output_paths(cfg)


if __name__ == "__main__":
    run()
