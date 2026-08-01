# Phase 2 — Research Projects, Built End to End

Phase 1 taught the algorithms. **Phase 2** is where we build complete research projects — each taken from an idea to a paper — following the same skeleton every time: **environment → reward → data → method → harness → an honest result.**

Every project ships with a lecture deck, a hand-illustrated companion book, a paper, and the full reproducible code in its folder here.

| # | Project | What you build | Code |
|---|---|---|---|
| 01 | **Socratic Alignment of Small Language Models** | Align a small LM to *withhold* the answer and guide with a question, scored by a verifiable leakage reward, an LLM judge, and real learning-gain. Nine recipes (SFT · DPO · KTO · ORPO · SimPO · GRPO · PPO) — the crossroads where Lecture 06's GRPO meets Lecture 05's PPO. | [`socratic-alignment/`](./socratic-alignment/) |
| 02 | Dream to Catch — an IRIS World Model | *(companion book + paper on the course site)* | — |
| 03 | Dreaming to Dodge — a first-person world model (VizDoom) | Evolve a controller entirely inside imagination and dodge fireballs. | [`dreaming-to-dodge/`](./dreaming-to-dodge/) |
| 04 | Grasping in the Dark — HIL-SERL (SAC + RLPD) | RL teaches a robot arm to grasp, from a reward it almost never sees. | [`grasping-in-the-dark/`](./grasping-in-the-dark/) |
| 05 | **Teaching Machines to Code (SWE-RL)** | Reinforcement learning for *software engineering*, from the basics through three projects: **Mini-SWE-RL** (bug-fixing on a laptop, 66.7% → 73.3%), **agentic RL on real code** (MBPP+ on Modal, real before/after fixes + paper), and **ECHO** (a terminal agent that learns a *world model for free*, ≈2× GRPO — runs in progress). The tests are the teacher. | [`swe-rl/`](./swe-rl/) |

## Project 01 · Socratic Alignment

- 📊 **Slides:** <https://rl-bootcamp-decks.vercel.app/lecture-p2-socratic/>
- 📖 **Companion book (14 illustrated chapters):** <https://rl-bootcamp-decks.vercel.app/book-socratic/>
- 📄 **Paper (PDF):** <https://rl-bootcamp-decks.vercel.app/pdfs/socratic-alignment-paper.pdf>
- 💻 **Reproduce it:** [`socratic-alignment/`](./socratic-alignment/) — see its README for the Modal run commands.

**The finding:** a clean, verifiable reward (GRPO) can still *lose* to a cheap offline preference method on a one-sided "don't reveal" constraint. Pure-contrast DPO learns to evade; the anchored offline methods (SFT/KTO/ORPO/SimPO) are the ones that actually learn to guide.

## Project 05 · Teaching Machines to Code (SWE-RL)

- 📊 **Slides:** <https://rl-bootcamp-decks.vercel.app/lecture-p2-teaching-machines-to-code/>
- 📖 **Companion book (16 illustrated chapters):** <https://teaching-machines-to-code.vercel.app>
- 📄 **Paper (PDF):** [`swe-rl/paper/swe-rl-ipr-paper.pdf`](./swe-rl/paper/) · also <https://swe-rl-ipr.vercel.app/assets/ipr-paper.pdf>
- 💻 **Reproduce it:** [`swe-rl/`](./swe-rl/) (deck + book source) · [Mini-SWE-RL](https://github.com/RajatDandekar/Mini-SWE-RL) · [`swe-rl-ipr/`](https://github.com/RajatDandekar/Mini-SWE-RL/tree/main/swe-rl-ipr)

**The idea:** the environment is the teacher. Nobody labels the right answer — a real environment (the *tests*, or a *terminal*) grades the model's attempts, and GRPO reinforces what worked. Shown three ways: on a laptop (Mini-SWE-RL), in the cloud on real code (agentic RL + paper), and inside a live terminal (ECHO's world-model-for-free, ≈2× GRPO, full runs in progress).

---

© Vizuara AI Labs · 2026 · RL in Production, Cohort 2026.
