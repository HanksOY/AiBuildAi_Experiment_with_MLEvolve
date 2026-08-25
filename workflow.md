# MLEvolve Workflow

How the pieces fit together, from launching the program to getting a notebook.

---

## The short version

Think of it like hiring a builder.

| Stage | Who | What they do |
|---|---|---|
| 1 | **Setup agent** | Intake desk. First run only. Gets your API key working, reads your data, drafts the job description. |
| 2 | **Plan mode** | The meeting before work starts. Asks questions, writes a plan, waits for your approval. |
| 3 | **The search** | The crew. Many workers trying different solutions in parallel. |
| 4 | **Main agent** | Phone line to the site. Stays open while they work. |

**The one connection that matters:** what you approve in stage 2 gets glued onto the
job description that every worker in stage 3 reads. Without that, the plan would be a
document you looked at once.

**The one rule that matters:** nobody is interrupted mid task. Say `stop` and they
finish what is in their hands, save it, then stop.

---

## Overall flow

```mermaid
flowchart TD
    A([python run.py]) --> B{API key found?}
    B -- no --> S[Setup agent<br/>verify key, read data, draft goal/eval]
    S --> C
    B -- yes --> C[Load config + task description]
    C --> D[Prepare workspace<br/>input / working / submission]
    D --> E[clean_task_desc<br/>an LLM rewrites your goal]
    E --> F{Plan mode enabled?}
    F -- no --> H
    F -- yes --> G[PLAN GATE]
    G -- you quit --> X([Exit, nothing run])
    G -- you approve --> H[Search loop]
    H --> I[Export notebook + print paths]
    I --> Z([Done])

    M[Main agent<br/>listens on your keyboard] -. commands .-> H

    style G fill:#1f6feb,color:#fff
    style H fill:#238636,color:#fff
    style M fill:#9e6a03,color:#fff
```

---

## Stage 2: the plan gate

Three phases. `/back` steps you to the previous one.

```mermaid
flowchart LR
    A[Show how your goal<br/>was rewritten] --> B[Questions<br/>3 to 5, with options]
    B --> C[Conflicts<br/>your answers vs config]
    C --> D[Plan<br/>revise until happy]
    D --> E[Settings review<br/>tune parameters to the plan]
    E --> F([Approved])

    D -. /back .-> B
    E -. /back .-> D

    style F fill:#238636,color:#fff
```

At every prompt you can type a command or plain text. The slash is optional.

| | |
|---|---|
| `/ask <text>` | answer my question, change nothing |
| `/change <text>` | request a change |
| `/back` | previous phase or question |
| `/skip` `/go` `/quit` `/help` | |

Anything unrecognised asks again. It never guesses.

---

## Stage 3: the search loop

One "step" picks an agent based on the state of the solution it is building on.

```mermaid
flowchart TD
    P{Parent solution?} 
    P -- fresh start --> D1[draft_agent]
    P -- broken --> D2[debug_agent]
    P -- working --> D3[improve_agent]
    P -- stuck --> D4[evolution_agent<br/>or fusion_agent]
    P -- root full --> D5[aggregation_agent]

    D1 & D2 & D3 & D4 & D5 --> C[agents/coder<br/>writes the actual Python]
    C --> R[code_review_agent]
    R --> X[Run it in a subprocess]
    X --> PR[result_parse_agent<br/>extract metric, pass or fail]
    PR --> L[data_leakage_agent<br/>only if score looks too good]
    L --> J[(journal)]
    J --> P

    style C fill:#8957e5,color:#fff
```

**None of the five agents write code.** Each builds a differently framed prompt and hands
it to `agents/coder`. What makes them different is what goes in the prompt: novelty
pressure for drafts, error context for debug, branch history for evolution.

---

## Stage 4: the main agent

Runs beside the loop, never inside it.

```mermaid
flowchart LR
    K([your keyboard]) --> T[reader thread]
    T --> Q[(command queue)]
    Q --> L[search loop<br/>drains every 1s]
    T -. free text .-> M[feedback model]
    M -.-> K

    style Q fill:#9e6a03,color:#fff
```

| Command | Effect |
|---|---|
| `status` | progress, best score, elapsed |
| `seed <path>` | start a draft from a file you edited |
| `stop` | finish running steps, save, exit |
| `abort` | kill training immediately, then exit |

Free text is answered by the model on the reader thread, so the search never waits on it.

---

## Which model does what

Two models are configured. Work is split by cost.

| Model | Used for |
|---|---|
| **code** (`agent.code`) | writing solutions, drafting questions and plans, reviewing code |
| **feedback** (`agent.feedback`) | reading results, classifying what you typed, answering your questions |
| **no model** | settings arithmetic, command parsing |

That last row is deliberate. The code model failed to notice that `exec.timeout` was nine
times larger than `agent.time_limit`, so those checks are plain arithmetic instead.

---

## What connects everything

- **`cfg`** — written by setup, tuned in the gate, read by every agent
- **`task_desc`** — the gate's output is the search's input
- **`ControlBus`** — the only channel open once the search starts
