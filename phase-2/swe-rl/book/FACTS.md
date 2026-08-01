# FACTS.md — the ground truth for *Teaching Machines to Code*

Every writer agent MUST obey this file. These are the real numbers, names, and claims from the three
projects. Do **not** invent figures. If a precise number is not here, describe the trend in words rather
than fabricate. When in doubt, be honest and modest — a sharp reader will check.

**Audience: absolute beginners.** Assume the reader does not know what a terminal is, what a test is, or
what "training a model" means. Define every term on first use, in plain words, with a real-life analogy.
Never assume prior ML knowledge.

**Global rule — do NOT mention "isomorphic perturbation" anywhere** except a single short mention as one
future-work idea in the final chapter (`where-this-goes-next`), framed generically as "a way to make the
reward harder to cheat." No details, no jargon, one paragraph maximum.

---

## SHARED CONCEPTS (used across all three projects)

- **Software engineering (SWE) task** = a concrete change to code with a definition of done. Canonical
  shape: there is a *bug* or a *feature request*, some *code*, and a *test* (an automatic check) that is
  currently failing; "done" = make the test pass without breaking the others. Real systems work from GitHub
  issues → a pull request (a proposed code change).
- **Terminal** (a.k.a. command line, shell) = a text-only window where you type commands to the computer
  (list files, run a program, run the tests) and it types text back. It is how developers and coding agents
  *do* things rather than just talk about them. Commands: `ls`, `cd`, `cat`, `python test.py`, `git`.
- **Test** = a small piece of code that checks another piece of code, e.g. `assert add(2,2) == 4`. If the
  check holds it "passes"; if not it "fails" (raises an error). Tests are the objective grader in this book.
- **Reinforcement learning (RL)** = learning by trial and reward. The learner (here, a language model) takes
  an **action**, gets a **reward** (a number: high = good), and adjusts to earn more reward next time.
  Analogy: learning to ride a bike / a dog learning a trick with treats. No human writes down the right
  answer; the environment scores attempts.
- **Reward in this book** = almost always "did the tests pass?" → 1.0 if all pass, 0.0 otherwise (binary).
  This is the recurring big idea: **the tests are the teacher.** Nobody labels the correct code by hand.
- **Language model / LLM** = a neural network trained to continue text; here it reads a bug + code and
  writes a fix. Models used are from the **Qwen2.5-Coder** and **Qwen3** families (open-weight coding models).
- **Parameters** = the tunable numbers inside the model; "1.5B" = 1.5 billion of them. More parameters =
  bigger, more capable, more expensive to run. "Training" = adjusting these numbers.
- **GRPO (Group Relative Policy Optimization)** = the RL algorithm used by ALL THREE projects. Recipe:
  for one problem, let the model try **G times** (a "group", e.g. G=8); score each try; compute each try's
  **advantage** = (its reward − the group's average reward), optionally divided by the group's standard
  deviation; then nudge the model to make above-average tries more likely and below-average ones less
  likely. Formula: `advantage_i = (reward_i − mean(rewards)) / std(rewards)`; `loss = −Σ advantage_i ×
  log P(response_i | prompt)`. KEY BEGINNER POINTS: (1) it needs a *value network*? NO — the group's own
  average is the baseline, which is what makes GRPO simpler than PPO. (2) If every try in a group gets the
  same reward (all pass or all fail), every advantage is 0 → no learning signal from that group. This is
  why "variance" / a mix of successes and failures matters (see the sweet spot).
- **GRPO is the algorithm behind DeepSWE** (an open coding agent that scores 42.2% → ~59% on SWE-bench
  Verified) and SWE-RL. Our projects use the *same algorithm* at smaller scale.
- **Modal** = a cloud service that rents GPUs and runs your code on them; used to train projects two and
  three on real hardware. **GPU** = the specialized chip that makes model training fast; **H100** = a
  high-end data-center GPU.
- **gVisor sandbox** = a safe, isolated container to run untrusted code (the model's code, or shell
  commands) without risk to the real machine.

---

## PROJECT ONE — Mini-SWE-RL  (github.com/RajatDandekar/Mini-SWE-RL)

Tagline: *train an RL-powered code-fixing agent from scratch on a laptop.* Same algorithm as DeepSWE
(42.2% on SWE-bench), miniaturized to run on Apple Silicon.

- **Model:** Qwen2.5-Coder-1.5B (1.5 billion parameters). **Hardware:** an Apple M4 Pro MacBook.
  **Training time:** ~30 minutes. Inference during rollouts uses **Ollama** (a local model server,
  ~0.3–0.8s/query); gradient updates use HuggingFace Transformers (needs the model's parameters).
- **Headline result (30 puzzles):** overall solve rate **66.7% (20/30) → 73.3% (22/30)**, i.e. **+6.7%**.
  - Hard puzzles (15): 73.3% → 73.3% (0.0%, unchanged).
  - Medium puzzles (15): 60.0% → 73.3% (**+13.3%**).
  - **7 puzzles newly solved** that the model had never solved before.
- **The environment — `CodeFixEnv`:** a tiny gym-style environment. `reset(puzzle_id)` gives a buggy
  function; `get_prompt()` formats it for the model; `step(fixed_code)` runs the tests and returns
  **reward = 1.0 if ALL tests pass else 0.0** (binary). This mirrors the real R2E-Gym environment (8,100
  real GitHub problems in Docker) — but small enough for a laptop.
- **The puzzle set — 45 puzzles across 3 difficulty files:** `puzzles.py` (15 easy), `puzzles_medium.py`
  (15 medium — Python gotchas), `puzzles_hard.py` (15 hard — algorithms/data structures).
  - Easy: the 1.5B model solved 100% (15/15) at temperature 0 → NO room for RL to learn.
  - Hard: 73.3% (11/15) at temperature 0.
  - Medium: designed for the "mixed" zone (30–70% solve rate) — this is where learning happens.
- **The "sweet spot" lesson (the project's key insight):** GRPO needs *variance* — some rollouts that
  succeed and some that fail on the SAME problem. All-pass groups (advantage 0) and all-fail groups
  (advantage 0) teach nothing. Testing every puzzle with 4 rollouts at temperature 0.8 gave: **Mixed
  (25–75% solve) = 18 puzzles (best learning signal); All solved = 9 (useless); All failed = 3 (useless).**
  The 18 "mixed" puzzles became the training set. Lesson: **puzzle difficulty must be calibrated to the model.**
- **Baseline failures (real examples to use):** `hard_scope_bug` (Python closure trap — a lambda captures
  a loop variable by reference), `hard_balanced_parens` (only handles `()`, ignores `[]` and `{}`),
  `hard_graph_cycle` (doesn't distinguish "visited" from "in the current DFS path"), `hard_eval_rpn`
  (Python `//` truncates toward −infinity, not toward 0), `med_class_shared_state` (a class-level mutable
  attribute shared across instances), `med_float_equality` (`0.1 + 0.2 != 0.3` in floating point),
  `med_generator_exhaustion` (a generator is consumed on the first pass), `med_zip_truncation` (`zip()`
  silently drops extra items). These are *great* teachable, real Python bugs.
- **Training config:** G = 8 attempts per puzzle, 15 puzzles × 8 = 120 rollouts (~2 min to collect),
  temperature 0.8 for exploration, ~10 epochs, `grpo_trainer_v2.py --no-ref`.
- **The point of the project:** DeepSWE uses Qwen3-32B on 64 H100 GPUs for 6 days; Mini-SWE-RL uses
  Qwen2.5-Coder-1.5B on one MacBook for 30 min. **The algorithm is identical** — only the scale differs.
  It is a complete, from-scratch version of the SWE-agent RL pipeline: environment (like R2E-Gym), agent
  (like DeepSWE), training (GRPO, like rLLM).

---

## PROJECT TWO — Agentic RL on real code  (the workshop's swe-rl project; NO isomorphic details)

The step up from toy puzzles to real programming tasks, run for real on cloud GPUs. This is the project
behind the workshop's research paper and its two websites.

- **Benchmark: MBPP+** (`evalplus/mbppplus`, 378 real small Python problems). Each problem has: a natural
  prompt ("write a function to…"), a few **visible tests** (the base asserts — what the reward sees), and a
  larger set of **hidden tests** (the "plus" suite — the honest ground truth the model never trains on),
  plus a gold reference solution. KEY IDEA for beginners: the visible tests are what you optimize; the
  hidden tests tell you whether the model *really* solved it or just satisfied the few visible checks.
- **Models:** Qwen2.5-Coder in three sizes — 0.5B, 1.5B, 7B (Instruct). Trained with **GRPO** (same as
  project one), reward = "did the visible tests pass?" (0/1). **LoRA** (a lightweight fine-tuning method
  that trains small adapter weights) lets a 7B model fit on one GPU.
- **The featured result — the 0.5B model** (chosen because it starts lowest, so it has the most room to
  improve and the clearest story): solved **44/100 → 51/100** held-out tasks; the learning curve **rises
  sharply from 0.44 to ~0.55–0.61 in the first steps, then plateaus** (the honest "learns fast then holds"
  shape). Trained 80 steps on an H100. **14 tasks flipped from fail → pass**, and headline framing =
  **"of the tasks it was failing, RL taught it to solve 14."** (Not a dramatic aggregate jump — an honest,
  real improvement.)
- **Why 0.5B and not 7B for the demo:** the 7B model already solved ~84% of tasks *before* any training
  (near the ceiling) → almost no room to show learning. Smaller models have more headroom → a clearer
  before/after story. This is a real, honest modeling lesson.
- **The five real before/after examples (all VERIFIED — the "after" passes the hidden tests too, so they
  are genuine fixes, not test-shaped guesses):**
  1. `find_Volume` (triangular prism): BEFORE `return length*width*height` (that's a *box*), AFTER
     `0.5*base*height*width` — the model had forgotten the ½ factor; a prism is half a box.
  2. `cube_Sum` (cube sum of first n even numbers): BEFORE the loop `range(2, n+1, 2)` stops too early
     (for n=2 it only reaches 2, missing 4), AFTER `range(2, 2*n+1, 2)` covers the first n even numbers.
  3. `tuple_to_dict`: BEFORE the loop reads `tup[i+1]` one step past the end of the tuple and crashes
     with an IndexError, AFTER it steps two at a time pairing each key with the next value.
  4. `max_product_tuple`: BEFORE returns the winning *pair* of numbers, AFTER returns their actual
     (absolute) *product* — it was returning the wrong thing.
  5. `find_Average_Of_Cube`: BEFORE divides by the wrong total, AFTER divides the sum of cubes by n —
     an average divides by how many items there are.
- **An honesty lesson worth telling (optional, beginner-friendly):** when curating examples, two "fixes"
  turned out to be fake — the model's function was already correct and only "failed" because it had appended
  its own buggy test lines. They were caught by *running the code*. Moral: verify by execution, and trust
  the hidden tests.
- **Infrastructure / real lessons:** trained on **Modal** (cloud GPUs, H100). A hard-won lesson: a naive
  background training job gets killed when the local session ends; the fix is to **deploy the job to the
  cloud and let it run fully server-side** (durable, independent of your laptop). Long RL runs must survive
  network drops and restarts. (Keep this concrete and gentle for beginners — the point is "training real
  models is a systems problem, not just a math problem.")
- Two public websites exist (a beginner "before/after" site at rl-teaches-code.vercel.app and a deeper
  research site at swe-rl-ipr.vercel.app) — you may mention that the beginner site shows these exact
  examples live, but do NOT go into the deeper site's specifics.
- **DO NOT** describe the isomorphic-perturbation reward here. If reward-hacking comes up, keep it to one
  sentence ("a model rewarded for passing tests can sometimes learn to game the tests") and defer the fix
  to the final future-work chapter.

---

## PROJECT THREE — ECHO: a world model, for free  (replication of arXiv 2605.24517, "ECHO: Terminal Agents Learn World Models for Free")

The most advanced project: a **terminal agent** (an agent that solves tasks by typing shell commands over
many steps) trained with RL, plus one extra idea that gives it a "world model" almost for free.

- **World model (beginner definition):** a model's internal ability to *predict what will happen next* in
  its environment. For a terminal agent: given the commands so far, predict what the terminal will print
  back. Analogy: before you flip a light switch, you already expect the light to come on — that expectation
  is a world model.
- **The ECHO idea in one sentence:** train the agent with GRPO as usual, but ADD a small second job —
  predict the **environment's response** (the terminal output tokens) — reusing the *same forward pass*.
  Learning to predict the computer's replies teaches the agent how the computer behaves, as a side effect
  of learning to act. "For free" = negligible extra compute (same forward pass, one extra loss term).
- **The loss (state it simply):** `L_ECHO = L_GRPO(actions) + λ · L_env(observations)`, with **λ = 0.05**.
  `L_env` = a standard next-token prediction loss (length-normalized cross-entropy) on the
  terminal-output ("observation") tokens. Beginners: "cross-entropy on the observation tokens" = "how
  surprised the model is by the terminal's reply; training reduces that surprise."
- **The benchmark: TerminalBench-2.0** — **exactly 89 real terminal tasks** (fix a bug, configure a tool,
  make a test pass — all by typing commands). "pass@1" = fraction solved on the first try.
- **Reported targets from the paper (frame carefully — see honesty note):** TB2 pass@1 roughly
  8B model 2.70 → 5.17, 14B 5.17 → 10.79, and an SFT-warm-started model 7.64 → 7.87. These are small
  absolute numbers (a handful of the 89 tasks).
- **THE HONEST FRAMING (must include, do not overclaim):** the **relative** result — ECHO roughly *doubles*
  GRPO's score in a controlled head-to-head (same everything, only the extra loss added) — is the real,
  high-confidence scientific claim. The **absolute** numbers (5.17, 10.79) are only a few tasks out of 89
  and depend on ~6,000 private training tasks we cannot reproduce, so a replication reports "near the
  ballpark, with a gap," not an exact match. Teach the *idea and the controlled A/B*, not a leaderboard.
- **Models & stack (open-weight, real):** base models Qwen3-8B / Qwen3-14B (and an SFT variant). Trainers:
  first **SkyRL** (FSDP), then a pivot to **prime-rl** (which ships ECHO natively and, crucially, is
  *fully asynchronous*). Training tasks come from open terminal-task corpora (e.g. endless-terminals,
  ~3,255 tasks). Rollouts run in **Modal sandboxes** (real, stateful gVisor terminals).
- **The engineering war story (great for beginners — "real ML is messy"):** the first trainer (SkyRL) is
  *synchronous* — the GPU inference engine sits idle for ~15 minutes during each long rollout (256 terminal
  episodes + grading + weight sync). A background watchdog timer (a safety timeout, ~600s) decided the idle
  engine had crashed and killed it — every configuration failed the same way. Root cause: *synchronous*
  training is a bad fit for *long, multi-step* agent rollouts. Fix: switch to **prime-rl**, which is
  **asynchronous** — the inference engines never sit idle — directly removing the idle-crash. Beginner
  moral: matching your training *system* to the *shape* of your task matters as much as the algorithm.
- A subtle correctness gotcha worth one sidenote: for ECHO's extra loss to actually do anything, the
  terminal-output tokens must sit *inside* the model's response span (so the model actually predicts them);
  a naive setup where observations go only into the *next* prompt makes ECHO silently do nothing. (Keep
  this to one sidenote; the lesson is "verify the extra loss is actually active.")
- **Status:** the ECHO idea is implemented and numerically verified active (the extra loss fires as
  expected); the small-scale smoke test passed; the full large runs are the ongoing work. Report this
  honestly — the project's value is the *idea* (a world model for free) and a *clean controlled comparison*,
  not a finished leaderboard number.

---

## FUTURE-WORK CHAPTER — the ONLY place to mention the "un-cheatable reward" idea
In `where-this-goes-next`, ONE short paragraph: "Because the reward is 'did the visible tests pass?', a
clever enough model can sometimes learn to satisfy those specific checks without truly solving the problem —
a mild form of gaming the reward. One research direction the workshop is exploring re-grades each attempt on
*altered but equivalent* versions of the tests, so an answer shaped to the exact visible test no longer
scores — making the reward much harder to cheat." No name, no equations, no more than a paragraph. Then move
on to the other future directions (longer multi-step tasks, whole real repositories, and agents that carry a
world model everywhere — tying back to ECHO).
