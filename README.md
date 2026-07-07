# RL in Production — Bootcamp Resources

The companion repository for the **RL in Production** intensive workshop by Vizuara AI Labs.

Course site: <https://rl-production.vizuara.ai>

---

## What's in here

| Path | Contents |
|---|---|
| [`prerequisites/`](./prerequisites/) | **Start here.** A self-paced primer (PDF) of the math + programming the course assumes. Read before Week 1. |
| [`lectures/`](./lectures/) | One folder per lecture. Each contains a syllabus outline, reading list, and links to the slide deck. |
| [`phase-2/`](./phase-2/) | **Research projects, built end to end** — each from an idea to a paper, with reproducible code. Starts with Project 01, Socratic Alignment. |
| [`code/`](./code/) | Hands-on, runnable implementations of every algorithm we cover. Pedagogical first — short, commented, single-file where possible. |
| [`resources/`](./resources/) | Curated reading lists, paper PDFs, and links to external talks. |

## How to use this repo as a student

0. **Before the course starts:** work through [`prerequisites/rl-prerequisites.pdf`](./prerequisites/) — especially Parts 1, 2, and 6 — and take the self-diagnostic at the end.
1. Read the lecture outline in `lectures/<n>-<title>/README.md` *before* attending.
2. Run the corresponding code in `code/<topic>/` *during or after* the lecture.
3. Work through the suggested papers from `resources/`.

## Lecture index

| # | Topic | Status |
|---|---|---|
| 01 | [Fundamentals — MDPs, value, Bellman, DP/MC/TD](./lectures/01-fundamentals/) | ✅ Live |
| 02 | Q-learning and DQN | 🟡 Upcoming |
| 03 | Policy gradients (REINFORCE → TRPO → PPO) | 🟡 Upcoming |
| 04 | Actor-critic and GAE | 🟡 Upcoming |
| 05 | RLHF — language meets RL | 🟡 Upcoming |
| 06 | DPO and direct preference optimization | 🟡 Upcoming |
| 07 | GRPO and reasoning RL | 🟡 Upcoming |
| 08 | Embodied RL and VLA models | 🟡 Upcoming |
| … | … | … |

## Phase 2 — research projects

Complete research projects built end to end — environment → reward → data → method → harness → an honest result. Each ships with a lecture deck, a hand-illustrated companion book, a paper, and full reproducible code. See [`phase-2/`](./phase-2/).

| # | Project | Code | Slides · Book · Paper |
|---|---|---|---|
| 01 | **Socratic Alignment of Small Language Models** — align a small LM to *withhold* the answer and guide with a question; nine recipes (SFT · DPO · KTO · ORPO · SimPO · GRPO · PPO) scored by a verifiable reward, an LLM judge, and real learning-gain. | [`phase-2/socratic-alignment/`](./phase-2/socratic-alignment/) | [Slides](https://rl-bootcamp-decks.vercel.app/lecture-p2-socratic/) · [Book](https://rl-bootcamp-decks.vercel.app/book-socratic/) · [Paper](https://rl-bootcamp-decks.vercel.app/pdfs/socratic-alignment-paper.pdf) |

## Code index

| Path | What it teaches |
|---|---|
| [`code/tic-tac-toe/`](./code/tic-tac-toe/) | DP vs MC vs TD value learning on a small, exhaustively-solvable MDP. The cleanest setting in which to *see* the three families of algorithms differ. |

---

## License

MIT. Use freely for teaching, learning, and research.

## Citing

If you build on this material, please cite the bootcamp:

```bibtex
@misc{vizuara_rl_in_production_2026,
  title  = {RL in Production — Cohort 2026},
  author = {Vizuara AI Labs},
  year   = {2026},
  howpublished = {\url{https://rl-production.vizuara.ai}},
}
```
