"""Inspector agent for the dashboard chat (mixed into assistants.Assistants). The checks are in inspector.py.

After an agent shows a result (settings, data profile, task draft, data plan, training plan), `inspect(project,
stage)` checks it on a background thread, so the user can read and approve without waiting. The result goes to
the chat as one Inspector message, and to project.json under "inspect" -> stage:
    {"digest", "msg", "status": "open" | "clean" | "sent" | "ignored" | "fixed", "note", "checks", "fix"}
A check that finishes after the result changed (another digest) is dropped.
"""

from __future__ import annotations

import threading
import time

from dashboard import inspector, wslexec
from dashboard.projects import Project

INSPECT_TIMEOUT_S = 180


class InspectAgent:
    inspect_mode = "thread"   # "thread" | "sync" (tests) | "off"

    def inspect(self, project: Project, stage: str) -> None:
        d = project.load()
        if not d.get("config", {}).get("inspector", True) or self.inspect_mode == "off":
            return
        dig = inspector.digest(stage, d)
        if self.inspect_mode == "sync":
            return self._inspect_run(project, stage, dig)
        threading.Thread(target=self._inspect_safe, args=(project, stage, dig), daemon=True,
                         name=f"inspect-{project.slug}-{stage}").start()

    def _inspect_safe(self, project: Project, stage: str, dig: str) -> None:
        try:
            self._inspect_run(project, stage, dig)
        except Exception as e:  # an Inspector failure must never disturb the other agents
            project.post("agent", f"The Inspector could not check {inspector.WHAT[stage]}: {e}", agent="inspect",
                         kind="error")

    def _facts(self, d: dict) -> list[dict]:
        path = d.get("data", {}).get("path")
        local = wslexec.to_local_path(path) if path else None
        return inspector.file_facts(d.get("data", {}).get("profile") or {},
                                    local if local is not None and local.is_dir() else None)

    def _inspect_run(self, project: Project, stage: str, dig: str) -> None:
        self.inspecting[(project.slug, stage)] = time.time()
        try:
            self._inspect_now(project, stage, dig)
        finally:
            self.inspecting.pop((project.slug, stage), None)

    def _inspect_now(self, project: Project, stage: str, dig: str) -> None:
        from dashboard.assistants import ask_claude, chat_model, json_from
        d = project.load()
        facts = self._facts(d) if stage in ("data", "plan") else []
        fix = None
        if stage == "setup":
            cpus = next((i.get("detail") for i in d.get("env", {}).get("items", []) if i.get("id") == "cpus"), None)
            checks = inspector.setup_rules(d, int(cpus) if str(cpus or "").isdigit() else None)
        elif stage == "data":
            checks = inspector.data_rules(d, facts)
        elif stage == "task":
            checks = inspector.task_rules(d)
        elif stage == "eda":
            checks = inspector.eda_rules(d)
        else:
            checks, fix = inspector.plan_rules(d, facts)
        llm_note = None
        if stage != "setup":
            said = [m["text"] for m in project.messages() if m.get("role") == "user"][-12:]
            try:
                answer = json_from(ask_claude(inspector.llm_prompt(stage, d, said, checks), chat_model(d),
                                              timeout=INSPECT_TIMEOUT_S))
                checks += inspector.parse_llm(answer)
            except Exception as e:
                llm_note = f"Only the rule checks ran: the Claude check failed ({str(e)[:160]})."
        if inspector.digest(stage, project.load()) != dig:
            return  # the result changed while we checked it; the new result gets its own check
        problems = [c for c in checks if not c["ok"]]
        record = {"digest": dig, "t": time.time(), "checks": checks, "fix": fix, "note": inspector.note(checks),
                  "status": "open" if problems else "clean", "msg": None}
        if stage == "setup" and not problems:
            return project.update(lambda x: x.setdefault("inspect", {}).update({stage: record}))  # nothing to say
        actions = []
        if problems:
            if fix:
                actions.append({"label": fix["label"], "action": "inspect.fix", "args": {"stage": stage},
                                "primary": True})
            target = inspector.TARGET[stage]
            from dashboard.assistants import AGENTS
            actions.append({"label": f"Send {len(problems)} {'issue' if len(problems) == 1 else 'issues'} to the "
                            f"{AGENTS[target]}", "action": "inspect.send", "args": {"stage": stage},
                            "primary": not fix})
            actions.append({"label": "Ignore", "action": "inspect.ignore", "args": {"stage": stage}})
        msg = project.post("agent", inspector.summary(stage, checks, llm_note), agent="inspect", kind="inspect",
                           checks=checks, inspect_stage=stage, actions=actions or None)
        record["msg"] = msg["id"]
        project.update(lambda x: x.setdefault("inspect", {}).update({stage: record}))

    # buttons ----------------------------------------------------------------

    def _open_record(self, project: Project, stage: str) -> dict | None:
        rec = project.load().get("inspect", {}).get(stage)
        if not rec or rec.get("status") != "open":
            project.post("agent", "Those Inspector findings are already handled or out of date.", agent="inspect",
                         kind="error")
            return None
        return rec

    def _set_status(self, project: Project, stage: str, status: str) -> None:
        project.update(lambda x: x.setdefault("inspect", {}).get(stage, {}).update(status=status))

    def inspect_send(self, project: Project, stage: str) -> None:
        rec = self._open_record(project, stage)
        if rec is None:
            return
        self._set_status(project, stage, "sent")
        target = inspector.TARGET[stage]
        text = inspector.send_text(stage, rec["checks"])
        msg = project.post("user", text, agent=target)
        handler = {"setup": self.setup_message, "task": self.task_message, "eda": self.eda_message,
                   "plan": self.plan_message}[target]
        handler(project, text, msg["id"])

    def inspect_fix(self, project: Project, stage: str) -> None:
        rec = self._open_record(project, stage)
        if rec is None or not rec.get("fix"):
            return
        self._set_status(project, stage, "fixed")
        project.post("user", rec["fix"]["label"], agent="setup")
        self.apply_changes(project, rec["fix"]["changes"], "The Inspector's fix:")  # shows the plan again

    def inspect_ignore(self, project: Project, stage: str) -> None:
        if self._open_record(project, stage) is None:
            return
        self._set_status(project, stage, "ignored")
        project.post("user", "Ignore the Inspector's findings", agent="inspect")

    def inspect_message(self, project: Project, text: str, msg_id: str | None) -> None:
        """Any message to the Inspector checks the newest result again, now."""
        d = project.load()
        stage = next((s for s, ok in (("plan", d.get("plan", {}).get("plan")),
                                      ("eda", d.get("eda", {}).get("data_plan")),
                                      ("task", d.get("task", {}).get("draft")),
                                      ("data", d.get("data", {}).get("profile")),
                                      ("setup", True)) if ok), "setup")
        project.post("agent", f"I check {inspector.WHAT[stage]} again.", agent="inspect", reply_to=msg_id)
        self._inspect_run(project, stage, inspector.digest(stage, d))
        rec = project.load().get("inspect", {}).get(stage, {})
        if stage == "setup" and rec.get("status") == "clean":
            project.post("agent", "The settings pass all rule checks.", agent="inspect")


def open_inspections(d: dict) -> set[str]:
    """Message ids of the Inspector messages whose buttons still apply (newest per stage, not yet handled)."""
    return {r.get("msg") for r in d.get("inspect", {}).values() if r.get("status") == "open" and r.get("msg")}
