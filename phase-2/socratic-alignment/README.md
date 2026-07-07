# Socratic Alignment of Small Language Models

**Teaching small models to _withhold_ the answer and guide with a question — with a verifiable reward.**

Project 1 of the [Vizuara **RL in Production**](https://rl-bootcamp-decks.vercel.app/) workshop (Cohort 2026). This is the full, reproducible code for the project: a simulated classroom, a three-part scorecard for grading a tutor, the **SocraticBench** dataset, and **nine alignment recipes** (base · prompting · SFT · DPO · KTO · ORPO · SimPO · GRPO · PPO) — trained and evaluated end-to-end on [Modal](https://modal.com).

> **The finding.** Alignment run _backwards_: instead of teaching a model to answer, we align it to answer with a guiding question. A clean, verifiable leakage reward (GRPO) can still **lose** to a cheap offline preference method on a one-sided "don't reveal" constraint — the online RL methods (GRPO, PPO) barely move the policy on a small budget, pure-contrast DPO learns to _evade_, and the anchored offline methods (SFT/KTO/ORPO/SimPO) are the ones that actually learn to guide.

- 📊 **Slides:** https://rl-bootcamp-decks.vercel.app/lecture-p2-socratic/
- 📖 **Companion book (14 illustrated chapters):** https://rl-bootcamp-decks.vercel.app/book-socratic/
- 📄 **Paper (PDF):** https://rl-bootcamp-decks.vercel.app/pdfs/socratic-alignment-paper.pdf

---

## What's here

```
modal_apps/
  common.py         shared Modal images, Volumes, model registry
  gen_data.py       build the SocraticBench dataset (questions, chosen/rejected, keyphrases)
  train.py          train one LoRA adapter for one recipe (SFT/DPO/KTO/ORPO/SimPO/GRPO)
  ppo.py            classic RLHF: train a Bradley-Terry reward model, then PPO the policy
  evals.py          run_eval — the four-module scorecard (leakage · judge · learning-gain · tax)
sweep_workflow.js   Claude-Code Workflow: one agent per (method × model × seed), fan out + synthesize
sweep2_workflow.js  the β-sweep + follow-up configs
synthesize.py       aggregate every run into the before→after master table
results/            the master table + synthesis this project reported
figures/            paper figures
```

## Reproduce it

You need a [Modal](https://modal.com) account (`pip install modal && modal setup`) — every GPU job runs there, and each returns plain JSON so you (or an agent) can read the numbers straight back.

```bash
# 1. Build the dataset (~1,300 preference pairs) on Modal
modal run modal_apps/gen_data.py --target 1300

# 2. Train one recipe (one LoRA adapter)
modal run modal_apps/train.py --method dpo --model qwen0.5b --run-id dpo_qwen0.5b_s0

# 3. Classic RLHF baseline (reward model + PPO)
modal run modal_apps/ppo.py --model qwen0.5b --run-id ppo_qwen0.5b_s0

# 4. Aggregate all runs into the master table
python synthesize.py
```

To run the **whole 71-config sweep** the way the project did — one agent per config, in parallel — use the Claude-Code Workflow tool with `sweep_workflow.js`.

## Models & recipes

- **Models:** Qwen2.5-0.5B / 1.5B, SmolLM2-1.7B, SmolLM2-360M (edit the registry in `modal_apps/common.py`).
- **Recipes:** `sft · dpo · kto · orpo · simpo · grpo` via `train.py`; `ppo` via `ppo.py`; plus the `base` and `prompting-only` baselines in `evals.py`.
- **Eval:** every model is scored on the held-out **SocraticBench** with the same four modules — a model-free leakage rule, a Qwen2.5-7B Socratic judge, a simulated-student learning-gain, and standard benchmarks (ARC / TruthfulQA) for the alignment tax.

## Headline numbers (Qwen2.5-0.5B, judge 0–100 / leakage rate)

| recipe | judge before→after | leakage before→after |
|---|---|---|
| SFT   | 57.4 → 74.7 | 0.43 → 0.07 |
| KTO   | 57.4 → 75.4 | 0.43 → 0.12 |
| ORPO  | 57.4 → 74.1 | 0.43 → 0.07 |
| SimPO | 57.4 → 73.8 | 0.43 → 0.09 |
| DPO   | 57.4 → 44.5 (**regresses**) | 0.43 → 0.03 |
| GRPO  | 57.4 → 47.5 (barely moves) | 0.43 → 0.27 |

Full table: [`results/master_table.md`](results/master_table.md). Every number comes from a real GPU run — see [`results/SYNTHESIS.md`](results/SYNTHESIS.md).

---

© Vizuara AI Labs · 2026 · RL in Production, Cohort 2026.
