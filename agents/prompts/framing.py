"""Task framing: how prompts describe the role, the goal, and the required outputs.

Upstream MLEvolve was written for Kaggle / MLE-Bench: the agent is a "Kaggle
grandmaster", success is a leaderboard, and every solution must write
./submission/submission.csv. Two config keys make that framing switchable:

    task.profile             kaggle (upstream wording) | generic (neutral wording)
    task.require_submission  true (upstream) | false (the printed validation score
                             is the result; a prediction file is optional)

Prompt text is written once, in its Kaggle form, and passed through
Framing.adapt(). With the defaults adapt() returns its input unchanged, so
upstream prompts stay byte-identical. Otherwise it applies the rewrite tables
below. tests/test_framing.py fails if a Kaggle phrase reaches a generic prompt.
"""

from dataclasses import dataclass

KAGGLE = "kaggle"
GENERIC = "generic"
PROFILES = (KAGGLE, GENERIC)

# (Kaggle text, generic text). Applied in order, so a longer phrase must come
# before any shorter phrase it contains.
GENERIC_REWRITES = [
    # Role lines
    ("You are a Kaggle grandmaster attending a competition, an expert in writing clean, efficient, "
     "and competition-winning Python code for ML tasks.",
     "You are an expert machine learning engineer, skilled at writing clean, efficient, "
     "high-performing Python code for ML tasks."),
    ("You are a Kaggle grandmaster attending a competition.",
     "You are an expert machine learning engineer."),
    ("🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.",
     "You are an expert machine learning engineer building a strong, well-validated solution."),
    ("with the quality expected of a Kaggle Grandmaster",
     "with the quality expected of an expert ML engineer"),
    # Goal lines
    ("✓ Compete for TOP performance, not trivial baselines",
     "✓ Aim for TOP performance on the task's metric, not trivial baselines"),
    ("Your solution will be evaluated on a real leaderboard.",
     "Your solution will be judged by the evaluation the task specifies."),
    ("Now, let's begin the competition. ", "Now, let's begin. "),
    ("for a competitive solution", "for a strong solution"),
    ("🎯 As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance.",
     "🎯 Make MEANINGFUL improvements that raise the task's validation metric."),
    ("required by the leaderboard", "required by the task"),
    ("distilled from the kaggle award-winning solutions", "distilled from prior winning ML solutions"),
    ("machine learning competition task", "machine learning task"),
    # Metric wording
    ("the official competition task and official evaluation metric",
     "the task as specified and its evaluation metric"),
    ("the official metric definition", "the task's metric definition"),
    ("the official specification", "the task specification"),
    # Data preview
    ("**COMPETITION DATA STRATEGY - I will READ CAREFULLY**", "**DATA STRATEGY - I will READ CAREFULLY**"),
    ("In competitions, 'validation' files", "In many datasets, 'validation' files"),
    # MLE-Bench environment
    ("applies **ONLY to submission evaluation after code generation** (for mle-bench test set).",
     "applies **ONLY to final evaluation after code generation**, if the task mentions one."),
    ("Code execution MUST complete within 9 hours (hard limit)",
     "Code execution MUST complete within the max execution time per run above (hard limit)"),
    ("Execution time: 9 hours available", "Execution time: up to the run's per-execution timeout"),
]

# Applied when task.require_submission is false: the prediction file becomes
# conditional on the task asking for one.
OPTIONAL_SUBMISSION_REWRITES = [
    ("**2. Generate submission.csv**", "**2. Prediction File (only if the task asks for one)**"),
    ("• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
     "• If the task requires predictions on a test set, save them to `./submission/submission.csv` "
     "(NOT ./working/submission.csv)"),
    ("• Content: Model predictions on ALL test samples",
     "• Otherwise no prediction file is needed; the printed validation score is the result"),
    ("□ Did I generate submission.csv in correct path with ALL test predictions?",
     "□ If the task asks for a prediction file, did I save ALL test predictions to "
     "./submission/submission.csv?"),
    ("**Submission File Location**: Must save the submission to `./submission/submission.csv`.",
     "**Submission File Location**: If the task asks for a prediction file, it goes in "
     "`./submission/submission.csv`. Do not flag a missing file otherwise."),
    ("NEVER remove a required output: `./submission/submission.csv`, the metrics files,",
     "NEVER remove a required output: any prediction file the task asks for, the metrics files,"),
    ("Complete test inference and submission.csv generation",
     "Complete test inference and submission.csv generation, if the task asks for a prediction file"),
    ("and saves submission.csv", "and saves submission.csv if the task asks for a prediction file"),
    (", perform test inference, and save `submission.csv` to `./submission/` directory.",
     ". Only if the task asks for a prediction file, perform test inference and save `submission.csv` "
     "to `./submission/` directory."),
    ("and generate submission file.", "and generate the submission file if the task asks for one."),
]


@dataclass(frozen=True)
class Framing:
    profile: str = KAGGLE
    require_submission: bool = True

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"task.profile must be one of {PROFILES}, got {self.profile!r}")

    @property
    def is_upstream(self) -> bool:
        return self.profile == KAGGLE and self.require_submission

    def adapt(self, text: str) -> str:
        if self.is_upstream:
            return text
        if self.profile == GENERIC:
            text = _apply(text, GENERIC_REWRITES)
        if not self.require_submission:
            text = _apply(text, OPTIONAL_SUBMISSION_REWRITES)
        return text

    def adapt_lines(self, lines: list) -> list:
        if self.is_upstream:
            return lines
        return [self.adapt(line) if isinstance(line, str) else line for line in lines]


UPSTREAM = Framing()


def _apply(text: str, rewrites) -> str:
    for old, new in rewrites:
        text = text.replace(old, new)
    return text


def framing_of(cfg) -> Framing:
    """Framing from cfg.task; configs without a task block get upstream framing."""
    task = getattr(cfg, "task", None)
    if task is None:
        return UPSTREAM
    return Framing(
        profile=getattr(task, "profile", KAGGLE),
        require_submission=bool(getattr(task, "require_submission", True)),
    )
