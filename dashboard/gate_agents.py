"""EDA agent, Plan agent and first-draft review for the dashboard.

These are the three steps where the terminal run waits for the user
(agents/eda_agent.py, agents/plan_mode.py, agents/draft_review.py). A run
started from the dashboard has no terminal, so:

* EDA and Plan run here, before launch. Each step calls the terminal gates' own
  functions in WSL (dashboard/gates.py), so the prompts and models are the same.
  The run then starts with the approved text (plan_mode.preapproved=True).
* Draft review stays inside the run, which writes logs/draft_review.jsonl and
  reads the answers this module appends to logs/draft_review_answers.jsonl.

Mixed into assistants.Assistants; uses its store, runs_root and helpers.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from dashboard import envcheck, wslexec
from dashboard.gates import RESULT_MARK
from dashboard.projects import Project

GATE_TIMEOUT_S = {"eda_report": 2700, "plan_start": 900, "plan_draft": 900, "eda_plan": 900}
APPROVE_WORDS = re.compile(r"^\s*(ok(ay)?|yes|yep|approve[ds]?|accept(ed)?|looks good|lgtm|go( ahead)?|"
                           r"fine|good|confirm(ed)?)\b[\s.!]*$", re.I)
OPTION_RE = re.compile(r"^\s*(?:option|opt|#)?\s*([1-9])\b", re.I)
REDO_RE = re.compile(r"\b(redo|rerun|re-run|new analysis|analy[sz]e again)\b", re.I)
DRAFT_REVIEW_FILE = "draft_review.jsonl"            # written only by the run
DRAFT_ANSWERS_FILE = "draft_review_answers.jsonl"   # written only by the dashboard
DRAFT_WORDS = {"accept": "accept", "run it": "accept", "go": "accept", "ok": "accept", "okay": "accept",
               "yes": "accept", "reject": "reject", "skip": "skip", "drop": "skip", "edited": "edited",
               "i edited": "edited", "reload": "edited"}

EDA_ACTIONS = [{"label": "Approve data plan", "action": "eda.approve", "primary": True},
               {"label": "Ask for a change…", "prefill": "@eda "},
               {"label": "Run a new analysis", "action": "eda.redo"}]


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def gate_overrides(c: dict) -> list[str]:
    """The project's settings as run overrides, so questions and plans see the real budget (as launch.sh does)."""
    return [f"agent.code.model={c['code_model']}", f"agent.feedback.model={c['feedback_model']}",
            "task.profile=generic", f"agent.steps={int(c['steps'])}",
            f"agent.time_limit={int(float(c['time_limit_h']) * 3600)}",
            f"agent.initial_drafts={int(c['initial_drafts'])}",
            f"agent.search.parallel_search_num={int(c['parallel'])}",
            f"exec.timeout={int(c['exec_timeout_min']) * 60}"]


def _q(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


class GateAgents:
    # WSL calls ----------------------------------------------------------------

    def gate(self, project: Project, cmd: str, **fields) -> dict:
        """One gate step in WSL (dashboard/gates.py). Raises RuntimeError on failure (eda_report returns it)."""
        d = project.load()
        c = d["config"]
        req = {"cmd": cmd, "project": wslexec.to_wsl_path(project.folder), "data": d["data"].get("path_wsl"),
               "overrides": gate_overrides(c), **fields}
        proc = wslexec.run_shell(f"cd {_q(envcheck.REPO_WSL)} && exec .venv/bin/python -m dashboard.gates",
                                 timeout=GATE_TIMEOUT_S.get(cmd, 600), input_text=json.dumps(req))
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith(RESULT_MARK)]
        if not lines:
            raise RuntimeError("the WSL step gave no answer: " + (proc.stderr or proc.stdout or "")[-600:])
        result = json.loads(lines[-1][len(RESULT_MARK):])
        if not result.get("ok") and cmd != "eda_report":
            raise RuntimeError("; ".join(result.get("errors") or ["unknown error"]))
        return result

    # EDA agent ----------------------------------------------------------------

    def eda_start(self, project: Project, echo: str | None = None) -> None:
        d = project.load()
        if not d["task"].get("accepted"):
            return project.post("agent", "Accept the task first; the analysis needs it.", agent="eda", kind="error")
        if echo:
            project.post("user", echo, agent="eda")
        project.update(lambda x: x.update(eda={"status": "running", "t": time.time()}, plan={}))
        project.post("agent", "I am writing an analysis script and running it on your data in WSL. This usually "
                     "takes 1 to 5 minutes. The script only reads the data; it trains nothing.", agent="eda",
                     kind="action")
        try:
            result = self.gate(project, "eda_report", task=d["task"]["draft"])
        except Exception as e:  # timeout or no answer: same handling as a failed analysis
            result = {"ok": False, "errors": [str(e)]}
        if not result.get("ok"):
            project.update(lambda x: x["eda"].update(status="failed", errors=result.get("errors")))
            errs = "\n".join(result.get("errors") or ["unknown error"])[-2500:]
            return project.post("agent", "The analysis failed. Last errors:\n\n```\n" + errs + "\n```", agent="eda",
                                kind="error", once=True,
                                actions=[{"label": "Try again", "action": "eda.redo", "primary": True},
                                         {"label": "Continue without EDA", "action": "eda.skip"}])
        report = result["report"]
        (project.folder / "EDA_report.md").write_text(report + "\n", encoding="utf-8")
        project.update(lambda x: x["eda"].update(status="report", report=report, feedback=[]))
        project.post("agent", f"I ran the analysis on your data ({result.get('seconds', '?')} s). Read the report, "
                     "then approve it or ask for a new analysis. It is saved as "
                     f"`projects/{project.slug}/EDA_report.md`.", agent="eda", card=report, long=True, once=True,
                     actions=[{"label": "Approve report", "action": "eda.accept", "primary": True},
                              {"label": "Run a new analysis", "action": "eda.redo"},
                              {"label": "Ask about the report…", "prefill": "@eda "}])

    def eda_accept(self, project: Project, echo: bool = True) -> None:
        if project.load().get("eda", {}).get("status") != "report":
            return project.post("agent", "There is no report waiting for approval.", agent="eda", kind="error")
        if echo:
            project.post("user", "Approve report", agent="eda")
        project.update(lambda x: x["eda"].update(status="plan"))
        self._eda_plan(project)

    def _eda_plan(self, project: Project, msg_id: str | None = None, lead: str | None = None) -> None:
        d = project.load()
        plan = self.gate(project, "eda_plan", task=d["task"]["draft"], report=d["eda"]["report"],
                         feedback=d["eda"].get("feedback") or [])["plan"]
        project.update(lambda x: x["eda"].update(data_plan=plan))
        project.post("agent", lead or "Here is the data plan, made from the report. Every agent in the run gets it "
                     "with the task. Approve it or tell me what to change.", agent="eda", card=plan, once=True,
                     reply_to=msg_id, actions=EDA_ACTIONS)
        self.inspect(project, "eda")

    def eda_approve(self, project: Project, echo: bool = True) -> None:
        eda = project.load().get("eda", {})
        if eda.get("status") != "plan" or not eda.get("data_plan"):
            return project.post("agent", "There is no data plan waiting for approval.", agent="eda", kind="error")
        if echo:
            project.post("user", "Approve data plan", agent="eda")
        (project.folder / "EDA_report.md").write_text(
            eda["report"] + "\n\n---\n\n## Confirmed data plan\n\n" + eda["data_plan"] + "\n", encoding="utf-8")
        project.update(lambda x: (x["eda"].update(status="approved", approved_t=time.time()), x.update(plan={})))
        project.post("agent", "Data plan approved. Next, the Plan agent prepares the training plan.", agent="eda",
                     kind="action")
        self.advance(project)

    def eda_skip(self, project: Project) -> None:
        if project.load().get("eda", {}).get("status") not in ("failed", "report", "plan"):
            return project.post("agent", "There is nothing to skip now.", agent="eda", kind="error")
        project.post("user", "Continue without EDA", agent="eda")
        project.update(lambda x: (x["eda"].update(status="skipped"), x.update(plan={})))
        project.post("agent", "OK, no data plan. The run works from the task alone.", agent="eda", kind="action")
        self.advance(project)

    def eda_message(self, project: Project, text: str, msg_id: str | None) -> None:
        from dashboard.assistants import ask_claude, chat_model
        d = project.load()
        eda = d.get("eda", {})
        status = eda.get("status")
        if status == "running":
            return project.post("agent", "The analysis is still running. I will post the report here.", agent="eda",
                                reply_to=msg_id)
        if REDO_RE.search(text) or status in (None, "failed"):
            if d["task"].get("accepted"):
                return self.eda_start(project)
            return project.post("agent", "I analyse the data after you accept the task.", agent="eda", reply_to=msg_id)
        if status == "report" and APPROVE_WORDS.match(text):
            return self.eda_accept(project, echo=False)
        if status == "plan":
            if APPROVE_WORDS.match(text):
                return self.eda_approve(project, echo=False)
            r = self.gate(project, "interpret_review", message=text, plan=eda.get("data_plan", ""))
            if r["intent"] != "question" or not r.get("reply"):
                project.update(lambda x: x["eda"].setdefault("feedback", []).append(text))
                return self._eda_plan(project, msg_id, lead="I changed the data plan. Approve it or tell me what "
                                      "else to change.")
            return project.post("agent", r["reply"], agent="eda", reply_to=msg_id)
        prompt = ("You are the EDA agent of the MLEvolve dashboard. Answer the question from the measured EDA report "
                  "and data plan below, in two to five sentences. Cite numbers from the report. Never invent numbers."
                  f"\n\n# EDA report\n{eda.get('report', '')[:14000]}\n\n# Data plan\n"
                  f"{eda.get('data_plan') or '(not written yet)'}\n\n# Question\n{text}")
        project.post("agent", ask_claude(prompt, chat_model(d)), agent="eda", reply_to=msg_id)

    # Plan agent ---------------------------------------------------------------

    def plan_start(self, project: Project) -> None:
        d = project.load()
        if d.get("eda", {}).get("status") not in ("approved", "skipped"):
            return project.post("agent", "Approve the data plan first (EDA stage).", agent="plan", kind="error")
        project.update(lambda x: x.update(plan={"status": "starting", "t": time.time()}))
        project.post("agent", "I am preparing the training plan: first the clean task text the agents get, then "
                     "a few questions for you.", agent="plan", kind="action")
        try:
            r = self.gate(project, "plan_start", task=d["task"]["draft"])
        except Exception as e:
            project.update(lambda x: x["plan"].update(status="failed"))
            return project.post("agent", f"I could not prepare the plan: {e}", agent="plan", kind="error", once=True,
                                actions=[{"label": "Try again", "action": "plan.restart", "primary": True}])
        questions = r.get("questions") or []
        project.update(lambda x: x["plan"].update(cleaned=r["cleaned"], diff=r.get("diff", ""), questions=questions,
                                                  answers=[], feedback=[]))
        if r.get("diff"):
            project.post("agent", "The run rewrites your task into a clean form for the agents. This is the change "
                         "(`-` removed, `+` added):", agent="plan", card="```\n" + r["diff"][-6000:] + "\n```",
                         long=True)
        if questions:
            project.update(lambda x: x["plan"].update(status="questions"))
            return self._ask(project, 0)
        project.post("agent", "I have no questions; the task is clear.", agent="plan")
        self._plan_draft(project)

    def _ask(self, project: Project, i: int) -> None:
        questions = project.load()["plan"]["questions"]
        q = questions[i]
        rec = q.get("recommended", 0)
        actions = [{"label": f"{j + 1}. {opt}" + (" (recommended)" if j == rec else ""), "action": "plan.answer",
                    "args": {"q": i, "o": j}} for j, opt in enumerate(q["options"])]
        actions.append({"label": "Answer in my own words…", "prefill": "@plan "})
        why = f"\n\n_{q['why']}_" if q.get("why") else ""
        project.post("agent", f"**Question {i + 1} of {len(questions)}.** {q['question']}{why}", agent="plan",
                     once=True, actions=actions)

    def plan_pick(self, project: Project, args: dict) -> None:
        plan = project.load().get("plan", {})
        try:
            i, j = int(args.get("q", -1)), int(args.get("o", -1))
        except (TypeError, ValueError):
            i = j = -1
        if plan.get("status") != "questions" or i != len(plan.get("answers", [])) or \
                not 0 <= j < len(plan["questions"][i]["options"]):
            return project.post("agent", "That question was already answered.", agent="plan", kind="error")
        option = plan["questions"][i]["options"][j]
        project.post("user", f"{j + 1}. {option}", agent="plan")
        self._record_answer(project, option)

    def _record_answer(self, project: Project, answer: str) -> None:
        d = project.update(lambda x: x["plan"]["answers"].append(
            [x["plan"]["questions"][len(x["plan"]["answers"])]["question"], answer]))
        i = len(d["plan"]["answers"])
        if i < len(d["plan"]["questions"]):
            return self._ask(project, i)
        self._plan_draft(project)

    def _plan_draft(self, project: Project, msg_id: str | None = None, lead: str | None = None) -> None:
        project.update(lambda x: x["plan"].update(status="drafting"))
        d = project.load()
        p = d["plan"]
        try:
            text = self.gate(project, "plan_draft", cleaned=p["cleaned"], answers=p.get("answers", []),
                             feedback=p.get("feedback", []))["plan"]
        except Exception as e:
            project.update(lambda x: x["plan"].update(status="review" if x["plan"].get("plan") else "failed"))
            return project.post("agent", f"I could not write the plan: {e}", agent="plan", kind="error", once=True,
                                actions=[{"label": "Try again", "action": "plan.restart", "primary": True}])
        project.update(lambda x: x["plan"].update(status="review", plan=text, final=None))
        self._post_plan(project, text, lead, msg_id)

    def _post_plan(self, project: Project, text: str, lead: str | None = None, msg_id: str | None = None) -> None:
        """The plan card with the current settings and the approve buttons, then the Inspector's check."""
        c = project.load()["config"]
        settings = (f"\n\n| Setting | Value |\n|---|---|\n| Steps | {c['steps']} |\n| Time limit | {c['time_limit_h']} h |"
                    f"\n| Parallel solutions | {c['parallel']} |\n| Review first drafts | "
                    f"{'on' if c.get('draft_review', True) else 'off'} |")
        project.post("agent", (lead or "Here is the training plan.") + " Nothing runs until you approve it. To change "
                     "a setting in the table, call `@setup`.", agent="plan", card=text + settings, once=True,
                     reply_to=msg_id,
                     actions=[{"label": "Approve plan and start training", "action": "plan.approve", "primary": True},
                              {"label": "Ask for a change…", "prefill": "@plan "},
                              {"label": "Back to the data plan", "action": "plan.back"}])
        self.inspect(project, "plan")

    def plan_message(self, project: Project, text: str, msg_id: str | None) -> None:
        d = project.load()
        p = d.get("plan", {})
        status = p.get("status")
        if status in (None, "failed"):
            if d.get("eda", {}).get("status") in ("approved", "skipped"):
                return self.plan_start(project)
            return project.post("agent", "I write the plan after the data plan is approved.", agent="plan",
                                reply_to=msg_id)
        if status in ("starting", "drafting"):
            return project.post("agent", "I am still working on it.", agent="plan", reply_to=msg_id)
        if status == "questions":
            q = p["questions"][len(p["answers"])]
            m = OPTION_RE.match(text)
            if m and 1 <= int(m.group(1)) <= len(q["options"]):
                return self._record_answer(project, q["options"][int(m.group(1)) - 1])
            r = self.gate(project, "plan_answer", message=text, question=q)
            if r["intent"] == "question" and r.get("reply"):
                return project.post("agent", r["reply"] + "\n\nAnswer with a button above or in your own words.",
                                    agent="plan", reply_to=msg_id)
            return self._record_answer(project, r.get("reply") or text)
        if status == "review" and re.fullmatch(r"\s*/?back\b.*", text, re.I | re.S):
            return self.plan_back(project, echo=False)
        if status == "review" and APPROVE_WORDS.match(text):
            return self.plan_approve(project, echo=False)
        r = self.gate(project, "interpret_review", message=text, plan=p.get("plan", ""))
        if r["intent"] == "question" and r.get("reply"):
            return project.post("agent", r["reply"], agent="plan", reply_to=msg_id)
        if d.get("run") and self.run_status_of(d["run"]) == "live":
            return project.post("agent", "The run that is going on keeps the approved plan. To steer it, send "
                                f"`@run guide {text[:200]}`.", agent="plan", reply_to=msg_id)
        project.update(lambda x: x["plan"].setdefault("feedback", []).append(text))
        self._plan_draft(project, msg_id, lead="I changed the plan.")

    def plan_back(self, project: Project, echo: bool = True) -> None:
        d = project.load()
        if echo:
            project.post("user", "Back to the data plan", agent="plan")
        if d.get("eda", {}).get("status") != "approved" or not d["eda"].get("data_plan"):
            project.update(lambda x: x.update(plan={}))
            return self.eda_start(project)
        project.update(lambda x: (x["eda"].update(status="plan"), x.update(plan={})))
        project.post("agent", "Back to the data plan. Approve it again or tell me what to change.", agent="eda",
                     card=d["eda"]["data_plan"], once=True, actions=EDA_ACTIONS)

    def plan_approve(self, project: Project, echo: bool = True) -> None:
        d = project.load()
        p = d.get("plan", {})
        if p.get("status") != "review" or not p.get("plan"):
            return project.post("agent", "There is no plan waiting for approval.", agent="plan", kind="error")
        if echo:
            project.post("user", "Approve plan and start training", agent="plan")
        eda = d.get("eda", {})
        final = self.gate(project, "compose", cleaned=p["cleaned"], answers=p.get("answers", []), plan=p["plan"],
                          data_plan=eda.get("data_plan") if eda.get("status") == "approved" else None)["text"]
        transcript = [f"# Plan — {d.get('name')}\n"]
        if p.get("diff"):
            transcript.append("## Task description rewrite\n\n```diff\n" + p["diff"] + "\n```\n")
        if p.get("answers"):
            transcript += ["## Clarifications\n"] + [f"**{q}**\n\n> {a}\n" for q, a in p["answers"]]
        transcript.append("## Approved plan\n\n" + p["plan"] + "\n")
        (project.folder / "plan.md").write_text("\n".join(transcript), encoding="utf-8")
        project.update(lambda x: x["plan"].update(status="approved", final=final, approved_t=time.time()))
        project.post("agent", "Plan approved. The run gets the clean task, your answers, this plan and the data plan.",
                     agent="plan", kind="action")
        self.run_launch(project)

    def _copy_plan_files(self, project: Project, run: str) -> None:
        """Put EDA_report.md and plan.md in the run folder, where the task text says the report is."""
        root = self.runs_root / run
        for name, dest in (("EDA_report.md", root / "EDA_report.md"), ("plan.md", root / "logs" / "plan.md")):
            src = project.folder / name
            try:
                if src.exists() and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
            except OSError:
                pass

    # Draft review (inside the live run) ---------------------------------------

    def pending_draft(self, run: str | None) -> dict | None:
        """The draft the run waits on, or None."""
        if not run:
            return None
        events = read_jsonl(self.runs_root / run / "logs" / DRAFT_REVIEW_FILE)
        done = {e.get("id") for e in events if e.get("type") == "result"}
        waiting = [e for e in events if e.get("type") == "draft" and e.get("id") not in done]
        return waiting[-1] if waiting else None

    def draft_answer(self, run: str | None, verdict: str, text: str = "", msg_id: str | None = None,
                     draft_id: str | None = None) -> bool:
        """Append an answer for the waiting draft. False when no draft (or not that one) is waiting."""
        pending = self.pending_draft(run)
        if pending is None or (draft_id and pending["id"] != draft_id):
            return False
        with open(self.runs_root / run / "logs" / DRAFT_ANSWERS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "id": pending["id"], "verdict": verdict, "text": text[:4000],
                                "msg": msg_id}) + "\n")
        return True

    def run_draft(self, project: Project, args: dict) -> None:
        verdict = str(args.get("verdict", ""))
        labels = {"accept": "Accept and run", "edited": "I edited the file", "skip": "Drop this draft"}
        if verdict not in labels:
            return project.post("agent", "Unknown draft answer.", agent="run", kind="error")
        msg = project.post("user", labels[verdict], agent="run")
        if not self.draft_answer(project.load().get("run"), verdict, msg_id=msg["id"], draft_id=args.get("id")):
            project.post("agent", "That draft is no longer waiting for review.", agent="run", kind="error",
                         reply_to=msg["id"])

    @staticmethod
    def draft_command(text: str) -> tuple[str, str]:
        """Map a chat message to a draft answer: (verdict, reason or question)."""
        t = text.strip()
        low = t.lower().rstrip(".!")
        for word in sorted(DRAFT_WORDS, key=len, reverse=True):
            if low == word or low.startswith(word + " ") or low.startswith(word + ":") or low.startswith(word + ","):
                rest = t[len(word):].lstrip(" :,-")
                return DRAFT_WORDS[word], rest
        return "ask", t


def draft_messages(run_dir: Path) -> list[dict]:
    """The run's draft review events as Run agent chat messages (for the project view)."""
    events = read_jsonl(run_dir / "logs" / DRAFT_REVIEW_FILE)
    done = {e.get("id"): e for e in events if e.get("type") == "result"}
    # The run (WSL) and the dashboard (Windows) clocks can differ by seconds. A reply or result always follows
    # the answer that caused it, so never let it sort before that answer.
    answered: dict[str, float] = {}   # draft id -> time of its last answer (for the result)
    asked: dict[str, float] = {}      # chat message id -> time of the question (for the reply)
    for a in read_jsonl(run_dir / "logs" / DRAFT_ANSWERS_FILE):
        answered[a.get("id")] = max(answered.get(a.get("id"), 0), a.get("t") or 0)
        if a.get("msg"):
            asked[a["msg"]] = a.get("t") or 0
    out = []
    for e in events:
        kind, rid, t = e.get("type"), e.get("id"), e.get("t") or 0
        if kind == "result" and rid in answered:
            t = max(t, answered[rid] + 0.5)
        elif kind == "reply" and e.get("reply_to") in asked:
            t = max(t, asked[e["reply_to"]] + 0.5)
        if kind == "draft":
            f = e.get("facts") or {}
            yes = lambda v: "yes" if v else "**not found**"  # noqa: E731
            card = (f"| Check | Result |\n|---|---|\n| Models | {f.get('models', '?')} |\n| Length | {f.get('lines', '?')} lines |\n"
                    f"| Cross-validation | {yes(f.get('cv'))} |\n| Prints the score | {yes(f.get('prints_score'))} |\n"
                    f"\nFile: `{e.get('path', '')}`")
            if e.get("plan"):
                card += "\n\n**Its plan**\n\n" + e["plan"][:2500]
            msg = {"id": f"dr-{rid}", "role": "agent", "agent": "run", "kind": "reply", "t": t, "long": True,
                   "text": f"**Draft {e.get('index')} of {e.get('total')} is written and waits for your review.** It "
                           "has not run yet, and the run's clock is paused while you read. Reply in your own words "
                           "to ask a question about it.", "card": card, "draft": rid}
            if rid not in done:
                msg["actions"] = [{"label": "Accept and run", "action": "run.draft",
                                   "args": {"id": rid, "verdict": "accept"}, "primary": True},
                                  {"label": "Show the code", "ui": "draft", "args": {"id": rid}},
                                  {"label": "Reject with a reason…", "prefill": "@run reject "},
                                  {"label": "I edited the file", "action": "run.draft",
                                   "args": {"id": rid, "verdict": "edited"}}]
            out.append(msg)
        elif kind == "reply":
            out.append({"id": f"dr-{rid}-r{t}", "role": "agent", "agent": "run", "kind": "reply", "t": t,
                        "text": e.get("text") or "", "reply_to": e.get("reply_to")})
        elif kind == "result":
            v = e.get("verdict")
            text = {"accept": f"Draft {e.get('index')} accepted. It runs now.",
                    "reject": f"Draft {e.get('index')} rejected. The run writes a new one with your reason.",
                    "abort": "The run stopped at the draft review."}.get(v, f"Draft review ended: {v}.")
            out.append({"id": f"dr-{rid}-done", "role": "agent", "agent": "run", "kind": "action", "t": t,
                        "text": text})
    return out
