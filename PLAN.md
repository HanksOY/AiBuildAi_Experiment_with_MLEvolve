# Experience Ledger on MLEvolve: build plan

## 0. How to use this plan

* Do the phases in the order of section 7. Do one phase at a time.
* Start each phase in plan mode. Read the related code. Write a short plan. Wait for approval. Then write the code.
* Write the unit tests first. Use the examples in section 5 as the test cases.
* A phase is complete only when all its acceptance criteria pass.
* Put all default values in config (section 5.11).

## 1. Goal and scope

A **tip** is one reusable idea for a model change, for example "use focal loss." The ledger keeps each tip as one **entry**. Each entry records every use of the tip and the result of each use.

The ledger ranks tips for the planner by relevance and by past results. It keeps this evidence across tasks.

MLEvolve memory finds tips that are similar to the problem. The ledger also shows which tips worked, how much they helped, what they cost, and in which situations.

**In scope:** the evaluation setup step, the metric registry, the ledger store, retrieval and menu, credit, six algorithm upgrades (sections 5.4 to 5.9), integration hooks, and an experiment harness.

**Out of scope:** changes to MCGS, new search actions, web search as a tip source, threshold repair, and clinical safety gates other than the floors in section 5.1.

## 2. What MLEvolve already gives (confirm in Phase 0)

* A global memory. It records the plan, code, metrics, and a success or failure label for each node. It uses BM25 and FAISS for retrieval.
* Two planning modes: base planning and memory enhanced planning.
* Three coding modes: single pass, stepwise multi agent, and SEARCH/REPLACE diff patching.
* Progressive MCGS with reference edges, stagnation detection, and cross branch fusion.
* Run outputs in `./runs/<timestamp>_<exp_id>/`.

## 3. Architecture

### Package layout

```
ledger/
  core/                 pure functions only, with unit tests
    registry.py         metric registry (5.2)
    gains.py            metric gains and combined gain (5.3)
    noise.py            paired bootstrap and outcome labels (5.4)
    value.py            value posterior and Thompson draws (5.5)
    situation.py        situation tags, pooling, carryover (5.6)
    menu.py             candidate scores and menu (5.7)
    credit.py           cited, undo, and delayed credit (5.8)
    maintenance.py      merge, split, and archive (5.9)
  setup.py              evaluation setup step (5.1)
  metrics.py            calculates rule metrics from saved predictions
  schema.py             dataclasses
  store.py              SQLite store, overlay for eval_frozen
  index.py              BM25 and embedding index, RRF relevance
  hooks.py              the only module that MLEvolve calls
  prompts.py            menu text, citation, marker, and quick run instructions
  undo.py               builds and runs undo variants
  tests/
```

### Data flow

**Task start.** The setup step fixes the evaluation rule (5.1). The rule does not change during the run.

**Each node:**
1. **Planning.** Hook H1 gets a menu. The ledger adds the menu as a separate section of the planner prompt.
2. The planner writes a plan. The last line of the plan is a `Sources:` line.
3. **Coding.** The coder tags each code change with the entry it implements. A flagged prompt section asks for this.
4. **Evaluation.** The node saves its validation predictions. Hook H3 gives the ledger the predictions, the GPU hours, and the status.
5. The ledger calculates the metrics, the gains, the outcome, and the credit. It updates the records. It makes new entries from new wins.

**Task end.** Hook H5 runs maintenance. In train mode, it saves the store. In eval_frozen mode, it discards the overlay.

## 4. Data model

**EvalRule:** `rule_id`, `rule_family`, `option` (`official` or `custom`), `output_type` (`binary`, `multiclass`, `regression`, `segmentation`, or `other`), `metrics` (names), `weights` (one per metric, total 1), `floors` (metric name and value), `source` (`file`, `prompt_block`, `user_answer`, or `default`).

**Entry:** `entry_id`, `family_id`, `title`, `description`, `code_ref`, `source_type` (`technique`, `reference_solution`, or `combination`), `source_ref`, `created_at`, `status` (`active` or `archived`), `aliases`.

**UseRecord:** `use_id`, `entry_id`, `task_id`, `node_id`, `parent_node_id`, `branch_id`, `rule_id`, `rule_family`, `situation`, `parent_metrics`, `child_metrics`, `metric_gains`, `combined_gain`, `floor_failed`, `gpu_hours`, `noise_sd`, `outcome` (`win`, `loss`, `inconclusive`, or `crash_pending`), `outcome_rule`, `credit_share`, `delayed_credit`, `is_exploration_slot`, `timestamp`.

**TaskProfile:** `task_id`, `modality`, `size_bucket`, `positive_share_bucket`, `output_type`, `rule_id`.

**MaintenanceLog:** each merge, split, and archive action, with the data to reverse it.

Keep all data in SQLite at `ledger.store_path`, outside `runs/`. Supply a JSON export command.

## 5. Algorithms

### 5.1 Evaluation setup

The ledger needs one fixed rule for each task. The rule tells which metrics count, how much each metric counts, and which floors apply.

**Rule sources.** Use the first source that exists:
1. The file `ledger_eval.yaml` in the task directory (format in appendix A).
2. A block that starts with `Evaluation:` in the task description (same format).
3. The user answer to the setup question. This applies only to interactive runs.
4. The default: Option A.

**Setup question.** Show it one time, before the search starts. Do not show it inside the planner.

> No evaluation method was specified. Choose one:
> A. Official metric only (<metric name>), weight 100%
> B. Custom: choose metrics and weights

**If the user selects B:**
* Show only the metrics that fit the output type:

| Output type | Allowed metrics |
|---|---|
| binary | AUC, log loss, sensitivity, specificity, precision, F1, accuracy, calibration error |
| multiclass | log loss, accuracy, macro F1, quadratic weighted kappa |
| regression | RMSE, MAE, RMSLE, R² |
| segmentation | Dice, Jaccard |

* Offer the preset "clinical screening" for binary tasks: sensitivity 40%, specificity 30%, AUC 20%, calibration error 10%, with a floor of sensitivity at least 0.80.
* The weights must have a total of 100%. If not, show an error and ask again.
* Ask the optional floor question: "Must a metric stay above or below a value?"

**MLE Bench warning.** For an MLE Bench task, show this text: "The MLE Bench grader uses only the official metric. Option B can lower the medal rate. Use Option A for benchmark comparisons."

**Unattended runs.** If no answer arrives in 120 seconds, use Option A. Log the source as `default`.

**Lock.** After setup, the rule does not change during the run. Store `rule_id` with each use record.

**Rule family.** All Option A rules have the family `official`. A custom rule gets its family from its metrics, weights, and floors. Two custom rules with the same metrics, weights, and floors have the same family.

**Threshold.** For threshold metrics (sensitivity, specificity, precision, F1, accuracy) on binary tasks, use a threshold of 0.5.

The setup step makes no LLM calls.

**Tests:**
* No file, no block, and no answer: the rule is Option A with the official metric, and the source is `default`.
* Weights of 40, 30, 20, and 9: the setup rejects them.
* Dice on a binary task: the setup rejects it.
* Two custom rules with the same metrics, weights, and floors: they have the same family.

### 5.2 Metric registry

Each metric has a name, a best value, the output types it fits, and a flag that shows if it needs a threshold. A metric with no known best value also has a direction.

| Metric | Best value |
|---|---|
| AUC, accuracy, sensitivity, specificity, precision, F1, macro F1 | 1 |
| Quadratic weighted kappa, R², Dice, Jaccard | 1 |
| Log loss, calibration error (10 bins), RMSE, MAE, RMSLE | 0 |

Phase 0 lists the official metrics of the selected MLE Bench tasks. Add each missing metric to the registry before a run. If the ledger cannot calculate an official metric, use the score that the node reports. In this case, the ledger cannot calculate paired noise (5.4).

### 5.3 Gains

**Metric gain.** For each metric in the rule:

metric gain = (child − parent) ÷ (best − parent)

Do not use an absolute value. The signs give the direction. The gain is the share of the distance to the best value that the child closed.

* If the parent has the best value, the metric gain is 0. Mark the use as "no headroom."
* If the metric has no known best value, use metric gain = d × (child − parent) ÷ max(|parent|, a small epsilon). The registry gives d: +1 if higher is better, −1 if lower is better.

**Combined gain.**
* If the child passes all floors: combined gain = Σ weight × metric gain.
* If the child fails one or more floors: combined gain = −Σ shortfall, with shortfall = |floor − child| ÷ |best − floor|. Set `floor_failed` to true.

**Value.** value = combined gain ÷ max(GPU hours, 0.1).

**Parent.** Use the primary parent only. Never use a reference node. Root nodes get no credit.

**Tests:**
* Option A, AUC 0.88 → 0.91, 3 GPU hours: metric gain 0.25, value 0.0833.
* Log loss 0.30 → 0.25: +0.1667. Log loss 0.30 → 0.35: −0.1667.
* AUC parent 1.0: gain 0, marked "no headroom."
* Clinical screening preset. Sensitivity 0.90 → 0.95 (+0.50). Specificity 0.80 → 0.78 (−0.10). AUC 0.88 → 0.91 (+0.25). Calibration error 0.08 → 0.06 (+0.25). Combined gain: +0.245.
* Floor of sensitivity at least 0.80, child sensitivity 0.75: shortfall 0.05 ÷ 0.20 = 0.25. Combined gain: −0.25.

### 5.4 Noise aware outcomes

This step needs validation predictions from the parent and the child on the same validation rows.

**Fixed split.** At task start, the ledger writes the validation row IDs to the task workspace. A flagged prompt section tells the coder to use this split. It also tells the coder to save predictions to `working/ledger_val_preds.csv`. The columns are `row_id`, `y_true`, and `y_pred`, plus a group ID column if the data has one. If MLEvolve already fixes the split or saves predictions, use that.

**Paired bootstrap.**
1. Make 1,000 samples of the validation rows, with replacement. If `ledger.noise.group_column` is set, sample groups, not rows.
2. In each sample, calculate all rule metrics for the parent and the child. Then calculate the combined gain.
3. `noise_sd` is the standard deviation of these combined gains.
4. Use at most 50,000 rows.

**Outcome.**
* If `floor_failed` is true: loss.
* Win if combined gain > 2 × noise_sd.
* Loss if combined gain < −2 × noise_sd.
* Otherwise: inconclusive.

**Fallbacks.** Use them in this order. Record the rule in `outcome_rule`.
1. Predictions exist, but the rows are different: noise_sd = √(sd_parent² + sd_child²).
2. No predictions, or the ledger cannot calculate a metric: basic rule. Win if combined gain ≥ 0.02. Loss if combined gain ≤ −0.02. Otherwise: inconclusive.

**Crashes.** Set the outcome to `crash_pending`. The next successful child on the same branch that cites the same entry resolves it. After 3 attempts without a resolution, set the outcome to inconclusive.

**Tests:**
* Option A, AUC 0.880 → 0.885, noise_sd 0.05: combined gain 0.0417 is less than 0.10. Outcome: inconclusive.
* Option A, AUC 0.880 → 0.910, noise_sd 0.05: combined gain 0.25. Outcome: win.
* Synthetic predictions with no true difference: at least 90% of trials are inconclusive.
* A child that fails a floor: loss.

### 5.5 Value and cost records

Each use gives a value v_i (5.3) with a weight w_i = credit_share_i × precision_i × carryover_i.

* precision_i = min(1, (s_ref ÷ noise_sd_i)²). s_ref is the median noise_sd in the ledger. If the noise is not known, use 0.5. For `crash_pending`, use 0.
* carryover_i comes from 5.6. It is 1 for uses from the current task.

**Pool combined values only inside one rule family.**

**Posterior mean:** m = (κ0 × μ0 + Σ w_i × v_i) ÷ (κ0 + Σ w_i), with κ0 = 2. μ0 is the family mean if the family has 3 or more weighted uses. Otherwise μ0 is 0.

**Posterior spread:** s = σ ÷ √(κ0 + Σ w_i). If Σ w_i ≥ 8, σ is the entry's own weighted standard deviation. Otherwise σ is the pooled standard deviation of all ledger uses. Until the ledger has 20 uses, σ is 0.1.

**Thompson draw:** θ ~ Normal(m, s). Use the task seed.

**Per metric records.** Each use also stores its metric gains. Pool them by metric name across all tasks and rule families. The menu evidence line and the optional need aware ranking (5.7) use them.

**Tests** (all weights 1, μ0 = 0, σ = 0.1):

| Entry | Values | m | s |
|---|---|---|---|
| View fusion | 0.095, 0.083, 0.071 | 0.0498 | 0.0447 |
| Pretraining | 0.006, 0.004 | 0.0025 | 0.0500 |
| Heavy augmentation | −0.100, −0.085, −0.065 | −0.0500 | 0.0447 |
| Untested entry | none | 0 | 0.0707 |

### 5.6 Situation records and carryover

**Tags for each use.** The ledger calculates them without an LLM call.
* `modality`: image, text, tabular, audio, time series, or other.
* `size_bucket`: less than 1k, 1k to 10k, 10k to 100k, or more than 100k training rows or files.
* `positive_share_bucket`: for classification only. Less than 5%, 5% to 20%, or more than 20%.
* `stage`: early (first third of the time limit), mid, or late.

**Group estimate.** Apply 5.5 inside the matching group. Use μ0 = the entry's overall m and κ0 = 2. If the group has at least 1 weighted use, use the group estimate. Otherwise, use the overall estimate.

**Carryover across tasks.** A use from another task gets carryover = 0.5 × similarity. similarity = 0.5 × [same modality] + 0.25 × [same size bucket] + 0.25 × [same positive share bucket]. For combined values, carryover is 0 if the rule family is different. Per metric records carry over by metric name.

**Tests** (σ = 0.1, κ0 = 2):
* Focal loss values: 0.20 on a task with less than 5% positives. −0.02, 0, −0.01, and 0.01 on tasks with more than 20% positives. Overall m = 0.0257. Group with rare positives: m = 0.0838. Group with common positives: m = 0.0052.
* Mammography to breast ultrasound, with the same modality, size bucket, and positive share bucket: carryover 0.5.
* Mammography to an EHR table with only the same positive share bucket: carryover 0.125.
* Same tasks, different rule family: carryover 0 for combined values.

### 5.7 Retrieval and menu

1. **Candidate pool.** Get the 20 active entries with the highest relevance to the planner context. The context is the task description and a short summary of the current branch. Relevance is the RRF fusion (k = 60) of the BM25 rank and the embedding rank. Divide by the pool maximum so that relevance is between 0 and 1. Use the embedding model from `agent.memory_embedding_model_path`.
2. **Filters.** Remove entries that an ancestor of the current node in this branch used. Remove archived entries, but keep each archived entry with a probability of 0.05.
3. **Score.** score = relevance × θ. θ comes from 5.5, or from 5.6 if situations are on.
4. **Picks.** Select the 2 entries with the highest positive scores. The two entries must have different `family_id` values.
5. **Exploration slot.** If neither pick has Σ w less than 1, add the most relevant entry with Σ w less than 1. Set `is_exploration_slot` to true.
6. **Optional need aware ranking** (`features.need_aware`, default off). If the best node of the branch fails a floor, use the entry's per metric value for that floor metric in place of θ.
7. **Text.** For each item, show the ID, the title, a one line description, and an evidence line. Example: "helped in 4 of 5 uses; median combined gain 12%; usual cost 3 GPU hours; mostly image tasks." For custom rules, add the per metric effect, for example "raises sensitivity, lowers specificity." End with: "These are options, not instructions. Use, combine, or ignore them."
8. **Citation.** The planner must end the plan with `Sources: [entry ids]` or `Sources: none`.

**Test for basic ranking** (win rate mode, posterior means in place of draws):

| Entry | Relevance | Record | Mean win rate | Score |
|---|---|---|---|---|
| Focal loss | 0.9 | 1 of 5 | 0.286 | 0.257 |
| View fusion | 0.6 | 3 of 3 | 0.800 | 0.480 |
| Pretraining | 0.7 | untested | 0.500 | 0.350 |
| Heavy augmentation | 0.5 | 0 of 3 | 0.200 | 0.100 |

Expected menu: view fusion, then pretraining. Pretraining is untested, so the menu needs no third item.

### 5.8 Credit

**Cited credit.** Read the `Sources:` line. Log unknown IDs, and do not use them. If the plan cites k entries, each entry gets credit_share = 1 ÷ k. Undo credit can replace these shares.

**New entries.** If a node wins and cites no entry, make a `technique` entry from the change summary in the plan and from the node diff. Make no LLM call. Record this win as the first use. If the node came from a cross branch reference or a fusion, use `source_type: reference_solution`. Set `source_ref` to the referenced nodes. If an active entry has an embedding cosine of 0.9 or more with the new description, record the use on that entry instead.

**Undo credit.**
* **Trigger:** the outcome is a win, the plan cites 2 or more entries, and at least one cited entry has Σ w less than 5.
* **Budget:** undo runs use at most 10% of the GPU time of the task. When the budget is used, do no more undo runs.
* **Marks:** a flagged coder prompt section tells the coder to tag each SEARCH/REPLACE block or code region with its entry ID. Phase 0 records the block format.
* **Variants:** for each cited entry j, rebuild the child code with the blocks of j set back to the parent version. If a variant fails to run, set undo to inconclusive for j.
* **Quick runs:** compare quick runs only with quick runs. If MLEvolve has a quick or debug mode, use it. If not, set `LEDGER_QUICK_RUN=1`. A flagged prompt section tells solutions to use fewer epochs or a data sample when this variable is set. Keep the quick score of each node in a cache.
* **Contribution:** measure all quick gains from the parent. d_j = gain(parent → child) − gain(parent → variant without j). Use the combined gain of 5.3.
* **Shares:** share_j = max(d_j, 0) ÷ Σ max(d, 0). If the sum is 0, keep equal shares.
* **Interaction check:** if |Σ d_j − gain(parent → child)| > 0.5 × gain(parent → child), keep equal shares. Then make a `combination` entry for the cited set, and record this use on it.

**Tests** (Option A, AUC, quick scores):
* Parent 0.880, child 0.910, without focal loss 0.905, without view fusion 0.885. d = 0.0417 and 0.2083. Shares: focal loss 0.167, view fusion 0.833. No interaction.
* Parent 0.880, child 0.910, without A 0.905, without B 0.905. Σ d = 0.0833, and the gain is 0.25. This is an interaction.

**Delayed credit.**
* When a node wins with combined gain g, go back along **primary** edges in the same branch, up to 3 steps. For an ancestor at distance t, each entry that the ancestor cited gets delayed_credit += 0.5^t × g × its credit share at that ancestor.
* The value of that use becomes (combined gain + delayed credit) ÷ GPU hours.

**Test** (Option A, AUC): attempt 5 cites "higher input resolution" and goes 0.880 → 0.881 (gain 0.0083). Attempts 6 and 7 cite nothing. They go 0.881 → 0.895 (gain 0.1176) and 0.895 → 0.905 (gain 0.0952). Total for the resolution entry: 0.0083 + 0.5 × 0.1176 + 0.25 × 0.0952 = 0.0910.

### 5.9 Maintenance (task end, train mode only)

* **Merge:** two active entries with a description cosine of 0.85 or more, posterior means with the same sign, and means less than one pooled σ apart. Merge into the older entry. Keep the other title as an alias. Move its use records.
* **Split:** the uses of an entry make two code variants, from the coder tags or from diff similarity. Each variant has Σ w of 3 or more. Their means are 2 or more pooled standard errors apart. Make two entries in the same family. Move the uses. Use the plan text of each variant for the names. Make no LLM call.
* **Archive:** Σ w is 8 or more, m + 1.28 × s is 0.01 or less, and the entry had no use in the last 2 tasks.
* Record each action in `MaintenanceLog`. Supply a command that reverses each action.

**Tests:**
* "Multi view fusion" (2 uses), "dual view fusion" (1), and "two view concatenation" (2), with similar means: they merge into one entry with 5 uses and 2 aliases.
* "Data augmentation" with a mild variant (6 uses) and a heavy variant (4 uses), with very different means: it splits into two entries.
* "Switch optimizer": 9 uses, m near 0, own standard deviation 0.005, no use for 2 tasks. Then s = 0.0015, and the upper bound is 0.0019. The entry is archived.

### 5.10 Menu as a set

Steps 2 and 4 of 5.7 apply this rule. Do not offer a tip that the branch already used. Do not offer two tips from the same family. Always keep the exploration slot.

### 5.11 Defaults

```yaml
ledger:
  enabled: false
  mode: train                     # train or eval_frozen
  store_path: /path/outside/runs/ledger.sqlite
  ranking: thompson_value         # relevance_only, thompson_winrate, or thompson_value
  features: {noise: true, undo: true, situations: true, delayed: true, maintenance: true, need_aware: false}
  eval:
    setup_timeout_s: 120
    threshold: 0.5
    calibration_bins: 10
    presets: {clinical_screening: {weights: {sensitivity: 0.4, specificity: 0.3, auc: 0.2, calibration_error: 0.1}, floors: {sensitivity: 0.80}}}
  retrieval: {pool_size: 20, rrf_k: 60, menu_picks: 2, exploration_slot: true, archived_keep_prob: 0.05}
  outcome: {basic_bar: 0.02}
  noise: {fixed_split: true, bootstrap_resamples: 1000, z: 2.0, group_column: null, max_rows: 50000}
  value: {prior_strength: 2, sigma_default: 0.1, pooled_sigma_min_uses: 20, own_sigma_min_weight: 8, min_gpu_hours: 0.1}
  situations: {pooling_k: 2, carryover_max: 0.5, sim_weights: {modality: 0.5, size: 0.25, positives: 0.25}}
  undo: {min_cited: 2, uncertain_weight: 5, max_gpu_share: 0.10, interaction_tolerance: 0.5}
  delayed: {decay: 0.5, max_depth: 3}
  maintenance: {merge_cosine: 0.85, split_min_weight: 3, split_z: 2.0, archive_min_weight: 8, archive_value_max: 0.01, archive_idle_tasks: 2}
  dedup_cosine: 0.9
```

## 6. Integration hooks

Each hook is one call into `ledger/hooks.py`. Phase 0 finds the location of each hook.

| Hook | Location | Call |
|---|---|---|
| H0 | Task start, before the first node | `setup_eval(task)` returns the locked EvalRule |
| H1 | Planner prompt assembly, both planning modes | `get_menu(context)` returns a prompt section |
| H2 | After the planner output | `parse_sources(plan_text)` |
| H3 | After node evaluation | `record_node(node, parent, gpu_hours, preds_path, status)` |
| H4 | Coder prompt assembly | adds the split, prediction, marker, and quick run instructions (flagged) |
| H5 | Task start and task end | `start_task(profile)` and `end_task()` |
| H6 | Before the first node | `prepare_split(task)` if `noise.fixed_split` is on |

## 7. Phases

### Phase 0: Reconnaissance (no code changes)

Write `docs/ledger/MLEVOLVE_MAP.md`. For each item, give the file, the function, and the line:
* The node class: metric, primary parent, reference edges, branch ID, status.
* Planner prompt assembly in base and memory enhanced modes.
* Coder prompt assembly, and the SEARCH/REPLACE block format.
* Memory writes and retrieval: record contents, BM25 and FAISS use, fusion method.
* Evaluation: how a node calculates its validation score, if it saves validation predictions, and if the validation split is the same for all nodes.
* Execution time, to calculate GPU hours.
* A quick or debug run mode, if one exists.
* Task metadata: metric name, data type, output type.
* What MLEvolve does when a task names no metric.
* The search tree log format in `runs/`.
* The official metrics of the selected MLE Bench tasks, and which metrics are missing from the registry (5.2).

Run one smoke task with the default settings. Record the run time and the artifacts. Propose a location for each hook H0 to H6. List the open questions.

**Acceptance:** the map is complete, the smoke run artifacts are listed, and the open questions are written down.

### Phase 1: Skeleton and safety net

Make the package layout, the `ledger:` config block with `enabled: false`, the schema, the SQLite store with overlay, the hooks as empty calls, an LLM stub for tests, and the regression test.

**Acceptance:**
* With the ledger off, the prompts for a fixed seed and the LLM stub are byte identical to upstream.
* With the ledger on and an empty store, the prompts are identical except for an empty menu section, or no section.
* The smoke task runs without change with the ledger off.
* The commands in `CLAUDE.md` are complete.

### Phase 2: Evaluation rule and gains

Make the setup step (5.1), the metric registry (5.2), the fixed split and the prediction saving (5.4), the metric calculation in `metrics.py`, and the gains (5.3).

**Acceptance:**
* The tests of 5.1, 5.2, and 5.3 pass.
* In an interactive smoke run with no rule, the setup question appears one time, before the search.
* In an unattended smoke run, the rule is Option A, and the log shows `default`.
* Each node of the smoke run has saved predictions, and the ledger calculates all rule metrics from them.

### Phase 3: Basic ledger (arm A2)

Make entry creation, the `Sources:` parser, and basic outcomes (5.4 fallback 2). Make Beta Thompson draws on the win rate: θ ~ Beta(wins + 1, losses + 1). Ignore inconclusive uses. Make the relevance index, the menu with the exploration slot, carryover at a fixed weight of 0.5, and the prompt sections. Include `ranking: relevance_only` for arm A1.

**Acceptance:**
* The ranking test in 5.7 passes.
* In the smoke task, menus appear in planner prompts.
* At least 90% of plans have a `Sources:` line that the parser can read. If not, change only the text of the citation instruction.
* The ledger writes use records.
* The store keeps its data across two smoke tasks in sequence.

### Phase 4: Noise aware outcomes

Make the paired bootstrap, the fallbacks, and the crash rules (5.4).

**Acceptance:** the tests of 5.4 pass. The log shows the share of nodes with paired noise. The bootstrap takes less than 5 seconds for each node at 50,000 rows.

### Phase 5: Value and cost records

Measure GPU hours. Make the posterior of 5.5, the per metric records, and `ranking: thompson_value`.

**Acceptance:** the tests of 5.5 pass. In the smoke task, menus show evidence lines based on value.

### Phase 6: Undo credit

Make the coder tags, the quick runs, the variants, the contributions, the shares, the interaction check, and the budget limit (5.8).

**Acceptance:** the undo tests of 5.8 pass. Variants of synthetic code with two tagged blocks build and run. The budget limit works.

### Phase 7: Situations and carryover

Make the tags, the group pooling, and the carryover from similarity (5.6). This replaces the fixed weight of 0.5.

**Acceptance:** the tests of 5.6 pass.

### Phase 8: Delayed credit

Make the delayed credit of 5.8, on primary edges only.

**Acceptance:** the delayed credit test of 5.8 passes. Reference edges never carry credit. The depth limit works.

### Phase 9: Maintenance

Make merge, split, archive, the log, and reversal (5.9).

**Acceptance:** the tests of 5.9 pass. After a reversal, the store is the same as before the action.

### Phase 10: Experiment harness

Make config files for each arm, task lists, a runner, metric extraction from `runs/`, statistics, and a Markdown report with tables and plots. Run the pilot first. Then run the full study (section 8).

**Acceptance:** the pilot runs from start to end and makes the report.

## 8. Experiment protocol

### Evaluation rule

* Use Option A for all MLE Bench runs. The grader uses only the official metric.
* Optional clinical pilot: use Option B with the clinical screening preset on an owner dataset. Use the same rule in all arms.

### Arms

| Arm | Configuration |
|---|---|
| A0 | Upstream MLEvolve, ledger off |
| A0b | Upstream with the fixed split and prediction saving only |
| A1 | Ledger menu, ranked by relevance only |
| A2 | Basic ledger (Phase 3) |
| A3 | A2 with noise aware outcomes |
| A4 | A3 with value and cost ranking |
| A5 | A4 with undo credit |
| A6 | A5 with situations and carryover |
| A7 | A6 with delayed credit |
| A8 | A7 with maintenance (full system) |

If the budget permits, also run A8 without noise, A8 without value, and A8 without undo.

### Tasks

* **Training tasks** build the ledger in `train` mode. **Evaluation tasks** use `eval_frozen`. The two sets do not overlap.
* Suggested sets: MLE Bench Lite (22 tasks) for evaluation, to compare with the MLEvolve ablations. Other MLE Bench tasks for training. Include medical imaging tasks where possible.
* **Pilot:** 6 training tasks, 6 evaluation tasks, a 3 hour time limit, 2 seeds, and arms A0, A2, and A8 only.
* **Full study:** 3 seeds for each arm and task.
* All arms use the same LLM, GPU type, time limit, and step budget.

### Metrics

* **Primary (MLE Bench):** medal rate, rate above the median, and valid submission rate on evaluation tasks.
* **Primary (clinical pilot):** combined score on a held out test split, each rule metric, and floor pass rate.
* **Cost:** GPU hours, LLM tokens, time to the best solution.
* **Ledger diagnostics:**
  * Menu uptake: the share of plans that cite a menu entry.
  * Exploration slot uptake.
  * Hit rate: the share of cited uses with the outcome win.
  * Inconclusive rate.
  * Predictive validity: the Spearman correlation between the m of an entry at retrieval time and its value on evaluation tasks.
  * Diversity: the effective number of branches (exp of the entropy of node counts across branches). Compare with A0. The ledger must not reduce exploration.

### Statistics

Give the mean and the standard error for each arm. Compare arms in pairs by task. Use a bootstrap confidence interval over tasks for the difference. Use a Wilcoxon signed rank test as a second check.

### Contamination checks

* Calculate a hash of the store before and after each evaluation task. The two hashes must be the same.
* Search the ledger tables and logs for test file paths and grader output fields. There must be none.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The planner ignores menus or does not cite sources | Measure menu uptake. Change only the citation text. Never make menus into instructions. |
| Custom weights hide a drop in one metric | Ask the optional floor question. Show per metric effects in the evidence line. |
| Option B on MLE Bench lowers medal rates | Show the warning. Use Option A for benchmark runs. |
| The registry has no entry for an official metric | Phase 0 lists the official metrics. Add them before a run. |
| The fixed split changes MLEvolve behavior | Arm A0b isolates this effect. |
| Quick runs do not agree with full runs | Report the agreement between quick and full gains when both exist. |
| Credit has noise | Noise aware outcomes first. Undo credit when a plan cites several entries. |
| The store becomes large | Maintenance, indexed tables, archive. |
| The bootstrap is slow | Use vector operations. Use at most 50,000 rows. |

## 10. Decisions for the owner before Phase 10

* The LLM backbone and the API endpoint.
* The GPU type and the number of GPUs.
* The time limit for each task, for the pilot and for the full study.
* The training and evaluation task lists, and the medical tasks to include.
* The number of seeds.
* The custom rule for the clinical pilot, if it is different from the preset.

## Appendix A: Evaluation block format

Use the same keys in `ledger_eval.yaml` and in an `Evaluation:` block of the task description.

```yaml
Evaluation:
  option: custom                  # official or custom
  output_type: binary
  metrics: {sensitivity: 40, specificity: 30, auc: 20, calibration_error: 10}
  floors: {sensitivity: ">= 0.80"}
```

For Option A, use only `option: official`.
