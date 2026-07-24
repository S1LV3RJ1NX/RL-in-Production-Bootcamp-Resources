---
theme: default
title: "Teaching Machines to Code — SWE-RL"
info: |
  RL in Production · SWE-RL Session · Vizuara AI Labs
  How reinforcement learning teaches a language model to do software
  engineering — from the basics, through three real projects.
class: text-center
colorSchema: light
drawings:
  persist: true
  presenterOnly: false
  syncAll: true
transition: slide-left
mdc: true
css: unocss
---

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --paper:   #F7F2E8;
  --paper-2: #F1EBDE;
  --panel:   #FDFAF3;
  --ink:     #1F1B16;
  --ink-2:   #4A4239;
  --ink-3:   #7C7165;
  --rule:    #D8CFBE;
  --accent-1: #8B3A3A; /* red — emphasis / policy / action */
  --accent-2: #3D5A4A; /* forest — value / critic / Q */
  --accent-3: #4A5D7E; /* slate — state / math */
  --accent-4: #8E6B2F; /* ochre — reward */
  --accent-5: #6B4E7E; /* plum — entropy / temperature / discount */
}

.slidev-layout {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-feature-settings: 'ss01', 'cv11';
  padding: 2.75rem 3.75rem !important;
  overflow-y: scroll !important;
  max-height: 100vh !important;
  height: 100vh !important;
}

.slidev-layout h1, .slidev-layout h2, .slidev-layout h3 {
  font-family: 'Fraunces', Georgia, serif !important;
  color: var(--ink) !important;
  font-weight: 500 !important;
  letter-spacing: -0.02em !important;
  line-height: 1.1 !important;
}
.slidev-layout h1 { font-size: 3.2rem !important; }
.slidev-layout h2 { font-size: 2.2rem !important; }
.slidev-layout h3 { font-size: 1.35rem !important; color: var(--ink-2) !important; font-weight: 600 !important; }
.slidev-layout h4 { font-family: 'Inter', sans-serif !important; color: var(--ink) !important; font-weight: 600 !important; font-size: 1.05rem !important; margin: 0 0 0.2rem 0 !important; }

.slidev-layout p, .slidev-layout li {
  font-size: 1.05rem !important;
  line-height: 1.55 !important;
  color: var(--ink) !important;
}
.slidev-layout em { font-style: italic; color: var(--ink-2); }
.slidev-layout strong { font-weight: 600; color: var(--ink); }

.slidev-layout code {
  font-family: 'JetBrains Mono', monospace !important;
  background: var(--paper-2) !important;
  color: var(--accent-1) !important;
  border: 1px solid var(--rule);
  padding: 0.1em 0.4em;
  border-radius: 3px;
  font-size: 0.92em;
}
.slidev-layout pre {
  background: var(--panel) !important;
  border: 1px solid var(--rule) !important;
  border-left: 3px solid var(--accent-1) !important;
  border-radius: 0 6px 6px 0 !important;
}
.slidev-layout pre code { background: transparent !important; border: none !important; color: var(--ink) !important; }
.shiki, .shiki span { color: var(--ink) !important; }

.slidev-layout blockquote {
  border-left: 3px solid var(--accent-1);
  background: var(--panel);
  padding: 0.9rem 1.2rem;
  border-radius: 0 6px 6px 0;
  font-style: italic;
  color: var(--ink-2);
}
.slidev-layout a { color: var(--accent-3) !important; text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 3px; }

.eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-3);
  font-weight: 500;
}
.lede {
  font-family: 'Fraunces', serif;
  font-weight: 300;
  font-size: 1.28rem;
  line-height: 1.5;
  color: var(--ink-2) !important;
  max-width: 44rem;
}
.callout {
  background: var(--panel);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent-1);
  border-radius: 0 6px 6px 0;
  padding: 1rem 1.3rem;
}
.callout-quiet {
  background: var(--paper-2);
  border-left: 2px solid var(--rule);
  padding: 0.85rem 1.15rem;
  border-radius: 0 4px 4px 0;
}
.rule-short { border: none; border-top: 1.5px solid var(--ink); width: 3rem; }
.rule { border: none; border-top: 1px solid var(--rule); }

.equation {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-size: 2.1rem;
  letter-spacing: -0.005em;
  color: var(--ink);
  text-align: center;
  padding: 1.5rem 0;
}
.equation-small {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-size: 1.4rem;
  color: var(--ink);
  text-align: center;
  padding: 0.5rem 0;
}
.v-state  { color: var(--accent-3); font-style: italic; }
.v-act    { color: var(--accent-1); font-style: italic; }
.v-reward { color: var(--accent-4); font-style: italic; }
.v-value  { color: var(--accent-2); font-style: italic; font-weight: 500; }
.v-gamma  { color: var(--accent-5); font-style: italic; }

.fig-caption {
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-size: 0.95rem;
  color: var(--ink-3);
  text-align: center;
  margin-top: 0.5rem;
}
.figimg {
  display: block; margin: 0 auto;
  border: 1px solid var(--rule);
  border-radius: 6px;
  background: #fff;
  box-shadow: 0 10px 30px -20px rgba(31,27,22,0.5);
}

.section-cover { text-align: left !important; }

.katex { color: var(--ink) !important; font-size: 1.05em; }
.katex .mord { color: var(--ink); }

.slidev-page-no,
.slidev-controls .slidev-icon-btn[title*="overview"],
.slidev-controls .slidev-icon-btn[title*="dark"],
.slidev-controls .slidev-icon-btn[title*="Menu"] {
  display: none !important;
}
</style>

<div class="text-left max-w-4xl" style="padding: 1.5rem 0;">

<div class="eyebrow mb-8">RL in Production · SWE-RL Session · Vizuara AI Labs</div>

# Teaching machines<br/>to code.

<p class="lede mt-6">
We spent the course learning reinforcement learning in simulators and games. Today we point it at <em>software engineering</em> — teaching a language model to fix real bugs and pass real tests. We build the idea from the ground up, then watch it work in <strong>three real projects</strong>: on a laptop, in the cloud, and inside a live terminal.
</p>

<hr class="rule-short mt-6" />

<div class="flex gap-16 mt-6 text-sm" style="color: var(--ink-3);">
  <div><div class="eyebrow mb-1">Duration</div>~Two hours</div>
  <div><div class="eyebrow mb-1">Format</div>From basics · three projects · live results</div>
  <div><div class="eyebrow mb-1">Instructor</div>Dr. Rajat Dandekar</div>
</div>

</div>

---
title: "The map"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Where we are going</div>

## Same idea, three times.

<p class="text-sm mt-1">Every project in this lecture is the <em>same move</em> — let a model try, let a real environment grade it, and reinforce what worked. What changes is the scale and the world.</p>

<div class="grid grid-cols-3 gap-6 mt-5">
<div class="callout-quiet">
<div class="eyebrow mb-1" style="color: var(--accent-1);">Project 1</div>
<h4>RL on your laptop</h4>
<p class="text-sm" style="margin:.3rem 0 0;">A bug-fixing agent, trained from scratch on a MacBook in <strong>30 minutes</strong>. Every gear visible.</p>
</div>
<div class="callout-quiet">
<div class="eyebrow mb-1" style="color: var(--accent-2);">Project 2</div>
<h4>Agentic RL, in the cloud</h4>
<p class="text-sm" style="margin:.3rem 0 0;">Real programming tasks, real GPUs, real before/after code. The workshop's own research project.</p>
</div>
<div class="callout-quiet">
<div class="eyebrow mb-1" style="color: var(--accent-5);">Project 3</div>
<h4>A world model, for free</h4>
<p class="text-sm" style="margin:.3rem 0 0;"><strong>ECHO</strong>: a terminal agent that learns to predict its computer — results training as we speak.</p>
</div>
</div>

<div class="callout mt-5">
<p class="text-sm" style="margin:0;"><strong>The one sentence to hold onto:</strong> nobody hand-writes the right answer. A real environment — the <em>tests</em>, or a <em>terminal</em> — grades the model's attempts, and reinforcement learning does the rest. <strong>The tests are the teacher.</strong></p>
</div>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 0 · Before the code</div>

# First — what is<br/>the job, exactly?

<p class="lede mt-5">Before we teach a machine to do software engineering, we should be able to say — in plain words — what the job <em>is</em>. No jargon yet: a broken function, a failing test, a definition of done a computer can check by itself.</p>

</div>

---
title: "What is a SWE task"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">The job</div>

## A software task is a change with a checkable goal.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/what-is-a-swe-task-1.png" style="width:100%;" />
<p class="fig-caption">A real, quiet bug: the code runs fine — it just charges the wrong price.</p>
</div>
<div>
<p class="text-sm">Imagine an online store. A tiny <strong>function</strong> (a named recipe the computer runs) is meant to take 20% off a price, but a human wrote <code>return price - discount</code> — subtracting the number 20, not 20 <em>percent</em>.</p>
<div class="callout-quiet mt-3">
<h4 style="color: var(--accent-1);">The most common kind of bug</h4>
<p class="text-sm" style="margin:0;">It does not crash or turn red. It runs happily and produces a <em>wrong answer</em>. So how would a machine know it is wrong — automatically?</p>
</div>
</div>
</div>

</div>

---
title: "The test is the teacher"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">The definition of done</div>

## The definition of "done" is a test.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/what-is-a-swe-task-2.png" style="width:100%;" />
<p class="fig-caption">"Done" = the failing test turns green. That check is the entire goal.</p>
</div>
<div>
<p class="text-sm">A <strong>test</strong> is a second, tiny piece of code that checks the first: <code>assert final_price(100, 20) == 80</code>. If it holds, it <em>passes</em>; if not, it <em>fails</em> loudly. Right now, our buggy code fails it.</p>
<p class="text-sm mt-2">So a software task, made concrete, is: some <strong>code</strong>, a <strong>test</strong> that is failing, and <em>done</em> = make it pass without breaking the others.</p>
<div class="callout mt-3">
<p class="text-sm" style="margin:0;">The goal is a number a computer checks by itself — <strong>pass or fail, 1 or 0</strong>. That is why RL fits code so well.</p>
</div>
</div>
</div>

</div>

---
title: "Real SWE, scaled up"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">The same shape, bigger</div>

## Real software engineering is this — scaled up and messier.

<img class="figimg mt-2" src="/figures/what-is-a-swe-task-3.png" style="max-width: 82%;" />

<p class="fig-caption">A real project is thousands of files. A bug arrives as an <strong>issue</strong>; a human or AI finds it, edits the code, and proposes a <strong>pull request</strong>; the project's tests decide if it is done. Same three parts — code, tests, "done = tests pass" — just bigger. The <em>finding</em> and <em>editing</em> is where an agent needs a terminal.</p>

</div>

---
title: "The terminal"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">The agent's hands</div>

## The terminal is how an agent *does* things.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/what-is-a-terminal-1.png" style="width:100%;" />
<p class="fig-caption">Words in, words out. You type a command; the computer replies.</p>
</div>
<div>
<p class="text-sm">A <strong>terminal</strong> (or command line) is a text window where you type commands — <code>ls</code> to list files, <code>python test.py</code> to run the tests — and the computer answers in text.</p>
<div class="callout-quiet mt-3">
<h4 style="color: var(--accent-4);">Giving an AI a terminal is like giving it hands</h4>
<p class="text-sm" style="margin:0;">It is the difference between an AI that <em>describes</em> a fix and one that actually runs the tests, reads the real error, edits the file, and checks that it worked.</p>
</div>
</div>
</div>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 0 · Before the code</div>

# Reinforcement learning,<br/>from scratch.

<p class="lede mt-5">One sentence: <em>try something, see if it worked, do more of what worked.</em> That is how you learned to ride a bike — and, it turns out, exactly how we will teach a model to code.</p>

</div>

---
title: "Two ways to teach"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">What makes RL different</div>

## RL learns from a *score*, not an answer key.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/rl-in-one-sitting-1.png" style="width:100%;" />
<p class="fig-caption">Copy-the-human needs someone to know the answer first. RL just needs a score.</p>
</div>
<div>
<p class="text-sm">Most model training <strong>copies</strong> human-labelled answers. Reinforcement learning is different: an <strong>agent</strong> takes an <strong>action</strong>, the <strong>environment</strong> hands back a <strong>reward</strong> (a number, high = good), and the agent adjusts to earn more next time.</p>
<p class="text-sm mt-2">No human writes the correct code. The environment scores attempts — and for code, that scorer already exists.</p>
</div>
</div>

</div>

---
title: "The loop, and why code fits"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">The loop</div>

## The RL loop — with the tests as the reward.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/rl-in-one-sitting-2.png" style="width:100%;" />
<p class="fig-caption">Model writes code → run the tests → reward → learn → repeat.</p>
</div>
<div>
<p class="text-sm">Plug code into the loop and every piece is concrete: the <span class="v-act">action</span> is a proposed fix, the <span class="v-state">environment</span> is the tests, and the <span class="v-reward">reward</span> is:</p>
<div class="equation-small mt-1">
<span class="v-reward">reward</span> = <span class="v-value">1.0</span> if all tests pass, else <span class="v-value">0.0</span>
</div>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">This is the recurring idea of the whole lecture: <strong>the tests are the teacher.</strong> Nobody hand-labels the code.</p>
</div>
</div>
</div>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 1 · The algorithm</div>

# GRPO — learning<br/>from a group of tries.

<p class="lede mt-5">One algorithm powers all three projects. <strong>GRPO</strong> — Group Relative Policy Optimization — is simple enough to say in a breath: let the model try the same problem several times, and reinforce the tries that beat the group's average.</p>

</div>

---
title: "GRPO: a group of tries"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">The recipe</div>

## Give the model the same problem eight times.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/grpo-learning-from-a-group-1.png" style="width:100%;" />
<p class="fig-caption">Eight attempts, each scored by the tests. The dashed line is the group average.</p>
</div>
<div>
<p class="text-sm">For one problem, sample a <strong>group</strong> of <code>G = 8</code> attempts (a little randomness — "temperature" — makes them differ). Score each with the tests: some 1s, some 0s.</p>
<p class="text-sm mt-2">GRPO never asks "was this good in the abstract?" It asks "was this attempt <strong>better or worse than its groupmates?</strong>"</p>
</div>
</div>

</div>

---
title: "The advantage, no critic"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">The math, gently</div>

## The advantage — and why there's no critic.

<div class="equation-small mt-1">
<span class="v-value">advantage<sub>i</sub></span> = ( <span class="v-reward">reward<sub>i</sub></span> − mean(rewards) ) / std(rewards)
</div>
<div class="equation-small">
loss = − Σ <span class="v-value">advantage<sub>i</sub></span> · log P( <span class="v-act">attempt<sub>i</sub></span> | prompt )
</div>

<p class="text-sm mt-3">In words: <strong>make above-average attempts more likely, below-average ones less likely.</strong> A passing attempt lands above the group mean → positive advantage → its probability goes up.</p>

<div class="grid grid-cols-2 gap-8 mt-3">
<div class="callout-quiet">
<h4 style="color: var(--accent-2);">No value network needed</h4>
<p class="text-sm" style="margin:0;">Older methods (PPO) train a separate "critic" to judge each attempt. GRPO's trick: the <strong>group's own average is the judge</strong>. Far less to build.</p>
</div>
<div class="callout-quiet">
<h4 style="color: var(--accent-1);">A warning it plants</h4>
<p class="text-sm" style="margin:0;">If all eight attempts get the <em>same</em> score, every advantage is <strong>0</strong> — no learning signal. Hold that thought.</p>
</div>
</div>

</div>

---
title: "The sweet spot"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">The learning signal</div>

## RL only learns from disagreement.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/the-sweet-spot-1.png" style="width:100%;" />
<p class="fig-caption">A flat group (all pass or all fail) teaches nothing. A mixed group is gold.</p>
</div>
<div>
<p class="text-sm">Because a flat group gives zero advantage, GRPO needs <strong>variance</strong>: some tries that succeed and some that fail on the <em>same</em> problem.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">A problem only teaches if it is <strong>just hard enough</strong> — not so easy the model always wins, not so hard it always loses. This "sweet spot" idea decides which problems we train on.</p>
</div>
</div>
</div>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 2 · Project one</div>

# RL on your laptop.<br/><span style="color: var(--accent-1);">Mini-SWE-RL.</span>

<p class="lede mt-5">A complete, from-scratch bug-fixing agent — small enough to train on a MacBook in half an hour, yet running the <em>same GRPO algorithm</em> as the systems that score on real GitHub bugs. Every moving part is visible.</p>

</div>

---
title: "The bug-fixing game"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">The environment</div>

## Turn "fix this bug" into a game.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/the-bug-fixing-game-1.png" style="width:100%;" />
<p class="fig-caption">CodeFixEnv: a buggy function in, a fix out, a 1 or 0 back.</p>
</div>
<div>
<p class="text-sm">The environment is <code>CodeFixEnv</code>: <code>reset()</code> hands over a buggy function, and <code>step(fix)</code> runs the tests and returns <strong>1.0 if all pass, else 0.0</strong>.</p>
<div class="callout-quiet mt-2">
<h4 style="color: var(--accent-2);">A miniature of the real thing</h4>
<p class="text-sm" style="margin:0;">This mirrors <strong>R2E-Gym</strong> (8,100 real GitHub bugs in Docker) — shrunk to a laptop. Model: <strong>Qwen2.5-Coder-1.5B</strong>, on an <strong>Apple M4 Pro</strong>.</p>
</div>
</div>
</div>

</div>

---
title: "The sweet spot, in data"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Calibrating difficulty</div>

## The sweet spot decided the training set.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/the-sweet-spot-3.png" style="width:100%;" />
<p class="fig-caption">Of 45 puzzles, only the 18 "mixed" ones carry a learning signal.</p>
</div>
<div>
<p class="text-sm">The project used 45 puzzles across three difficulties. The <em>easy</em> set was solved <strong>100%</strong> of the time — useless for RL. Testing every puzzle revealed three buckets:</p>
<ul class="text-sm mt-1">
<li><strong>Mixed (25–75% solved): 18 puzzles</strong> — the best signal</li>
<li>All solved: 9 — teaches nothing</li>
<li>All failed: 3 — teaches nothing</li>
</ul>
<p class="text-sm mt-1">The <strong>18 mixed puzzles</strong> became the training set.</p>
</div>
</div>

</div>

---
title: "The training loop"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">Try → score → nudge</div>

## The whole loop, in thirty minutes.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/training-on-a-laptop-1.png" style="width:100%;" />
<p class="fig-caption">Three steps, repeated ~10 times over the puzzles.</p>
</div>
<div>
<p class="text-sm">Each round: <strong>try</strong> (8 attempts per puzzle, collected via <code>Ollama</code> for speed), <strong>score</strong> (the tests hand back 1/0), <strong>nudge</strong> (GRPO advantage → one gradient step). That is it.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">120 attempts collected in ~2 minutes; ten passes over the mixed puzzles; done in about <strong>30 minutes</strong> on a laptop.</p>
</div>
</div>
</div>

</div>

---
title: "The result"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">The payoff</div>

## 66.7% → 73.3%, and seven new bugs.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/what-the-laptop-taught-us-1.png" style="width:100%;" />
<p class="fig-caption">A real, honest climb — biggest gains where there was room to grow.</p>
</div>
<div>
<p class="text-sm">Overall solve rate rose from <strong>66.7%</strong> to <strong>73.3%</strong>. On the medium puzzles — which had headroom — it jumped <strong>60.0% → 73.3%</strong>. Most tellingly, it <strong>solved 7 puzzles it never had before</strong>:</p>
<ul class="text-sm mt-1">
<li>a Python <strong>closure trap</strong> (a lambda capturing a loop variable)</li>
<li><strong>bracket matching</strong> beyond just <code>()</code></li>
<li>graph <strong>cycle detection</strong> done correctly</li>
</ul>
</div>
</div>

</div>

---
title: "Same algorithm as the labs"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">Why the small result is a big deal</div>

## The same algorithm as the frontier — 1/1000th the scale.

<img class="figimg mt-2" src="/figures/what-the-laptop-taught-us-4.png" style="max-width: 74%;" />

<p class="fig-caption"><strong>DeepSWE</strong> trains a 32-billion-parameter model on <strong>64 H100 GPUs for six days</strong> to score on real GitHub bugs. Mini-SWE-RL used a ~20× smaller model on <strong>one laptop for 30 minutes</strong> — and the GRPO loop on the page is <em>identical</em>. The idea is not locked inside a big lab.</p>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 3 · Project two</div>

# Agentic RL<br/>on real code.

<p class="lede mt-5">This is the heart of the lecture, and its name: <strong>agentic reinforcement learning</strong>. The model stops being a one-shot answerer and becomes an <em>agent</em> — it acts, sees what happens, and decides again, using real tools in a real environment. We build that machinery first, then run it for real on cloud GPUs and watch a model go from failing a task to solving it.</p>

</div>

---
title: "What makes RL agentic"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">The theory — what "agentic" means</div>

## From a one-shot answer to an agent.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agentic-single-vs-multi.png" style="width:100%;" />
<p class="fig-caption">Project 1 answered once. An agent acts, observes, and decides — many times.</p>
</div>
<div>
<p class="text-sm">In Project 1 the model did <strong>one thing</strong>: read the bug, write the fix, get graded. That is <strong>single-turn</strong> RL.</p>
<p class="text-sm mt-2"><strong>Agentic RL</strong> makes the model an <strong>agent</strong>: it takes <em>many</em> steps — run a command, read the error, edit a file, run the tests again — using real <strong>tools</strong> and reacting to whatever each one returns.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">This is the real shape of software work — and the core idea of this lecture. Everything on the next few slides is how that loop is actually built.</p>
</div>
</div>
</div>

</div>

---
title: "The agent-tool loop"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">The loop</div>

## Think, act, observe — and repeat.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agentic-react-loop.png" style="width:100%;" />
<p class="fig-caption">The model reasons, calls a tool, the environment runs it, and the result flows back.</p>
</div>
<div>
<p class="text-sm">One <strong>turn</strong> of an agent: it <span class="v-state">thinks</span>, then <span class="v-act">acts</span> by emitting a <strong>tool call</strong> (say, <code>run_sanctioned_tests</code>). The environment runs the tool and hands back an <strong>observation</strong>, which is appended to the model's context. Then it goes again.</p>
<p class="text-sm mt-2">This reason–act–observe pattern is often called <strong>ReAct</strong>. The loop runs up to <code>H</code> turns, then the agent calls <code>submit</code> and is graded.</p>
</div>
</div>

</div>

---
title: "The agent's tools"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">The tool surface</div>

## The agent's whole world: six tools.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agentic-tool-surface.png" style="width:100%;" />
<p class="fig-caption">Read, search, edit, run the visible tests, submit — all on a safe copy.</p>
</div>
<div>
<p class="text-sm">Our agent has exactly six tools: <code>bash</code>, <code>search</code>, <code>open</code>, <code>edit</code>, <code>run_sanctioned_tests</code>, and <code>submit</code>. It works on a private <strong>copy</strong> of the code, so it can experiment without risk.</p>
<div class="callout-quiet mt-2">
<h4 style="color: var(--accent-2);">Same contract, any scale</h4>
<p class="text-sm" style="margin:0;">These same six tools drive an agent over one small file — or, unchanged, over a whole GitHub repository. It is the interface production SWE-agents use.</p>
</div>
</div>
</div>

</div>

---
title: "How observations come back"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">How the observation comes back — the mechanism</div>

## The tool's output becomes the next thing the model reads.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agentic-observation-back.png" style="width:100%;" />
<p class="fig-caption">Dispatch the call, capture the output as text, trim it, append it to the chat.</p>
</div>
<div>
<p class="text-sm">When the model calls a tool, three things happen: the call is <strong>dispatched</strong> to the environment; the tool's raw output (a printout, an error, a file's contents) is <strong>captured as text</strong>; and that text is <strong>appended to the conversation</strong> as the observation the model reads next.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">One practical guard: observations are <strong>trimmed</strong> (here, to 4,000 characters) so a single <code>cat hugefile</code> can't flood the model's limited context window.</p>
</div>
</div>
</div>

</div>

---
title: "One trajectory, one reward"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Where the reward lives</div>

## A whole trajectory, graded by its final state.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agentic-trajectory-reward.png" style="width:100%;" />
<p class="fig-caption">Actions and observations alternate; the reward arrives only at the very end.</p>
</div>
<div>
<p class="text-sm">A full episode — a <strong>trajectory</strong> — is a chain of (action, observation) pairs ending in <code>submit</code>. The <span class="v-reward">reward</span> is <strong>sparse</strong>: it lands only on the final state (did the tests pass?), not on each step along the way.</p>
<p class="text-sm mt-2">So GRPO grades the <em>whole trajectory</em>: sample several complete runs of the same task, score each by its final reward, and reinforce the better runs — the same group-relative trick as before, now over entire trajectories.</p>
</div>
</div>

</div>

---
title: "Whose tokens get trained"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-5);">A detail that sets up Project 3</div>

## Only the agent's own tokens learn.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agentic-token-mask.png" style="width:100%;" />
<p class="fig-caption">Action tokens get the gradient; observation tokens are masked out.</p>
</div>
<div>
<p class="text-sm">A subtle but crucial point. The trajectory holds two kinds of tokens: the ones the <strong>model wrote</strong> (its actions) and the ones the <strong>tools returned</strong> (the observations). RL trains only the tokens the model actually generated — the observation tokens are <strong>masked out</strong>, because the model didn't write them.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Hold onto this. <strong>Project 3 (ECHO)</strong> asks: what if we <em>stopped</em> ignoring those observation tokens — and taught the model to predict them?</p>
</div>
</div>
</div>

</div>

---
title: "From puzzles to real tasks"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">From the general agent to a measurable task</div>

## Visible tests to train on, hidden tests to grade honestly.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/from-puzzles-to-real-code-1.png" style="width:100%;" />
<p class="fig-caption">MBPP+: each task has a few visible tests and many hidden ones.</p>
</div>
<div>
<p class="text-sm">To <em>train</em> that agent we need tasks we can grade by the thousand. <strong>MBPP+</strong> gives 378 real Python problems — a short-horizon slice of the loop (edit, run tests, submit) that still runs thousands of RL steps. Each has a prompt, a few <strong>visible tests</strong> (what the reward sees), and many <strong>hidden tests</strong> (the honest ground truth the model never trains on).</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">The visible tests are what you optimize; the hidden tests tell you if the model <em>really</em> solved it — or just satisfied the few checks it could see.</p>
</div>
</div>
</div>

</div>

---
title: "Why the small model"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">An honest modelling choice</div>

## We featured the 0.5B model — on purpose.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/before-and-after-rl-1.png" style="width:100%;" />
<p class="fig-caption">The 7B model was already near the ceiling; the 0.5B had room to grow.</p>
</div>
<div>
<p class="text-sm">We trained models from 0.5B to 7B parameters with GRPO. The <strong>7B already solved ~84%</strong> before any training — almost no room to show learning.</p>
<div class="callout-quiet mt-2">
<h4 style="color: var(--accent-2);">Headroom makes the story</h4>
<p class="text-sm" style="margin:0;">The <strong>0.5B</strong> started at <strong>44%</strong> — far more room to improve, and far more real before/after examples to show. So we feature it.</p>
</div>
</div>
</div>

</div>

---
title: "Before and after"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">The heart of the project</div>

## Before RL it failed. After RL it solves it.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/before-and-after-rl-2.png" style="width:100%;" />
<p class="fig-caption">Real code the same model wrote — before training vs after.</p>
</div>
<div>
<p class="text-sm">Five genuine fixes RL taught the 0.5B model (every "after" also passes the <em>hidden</em> tests, so they are real fixes, not test-shaped guesses):</p>
<ul class="text-sm mt-1">
<li><code>find_Volume</code> — forgot the ½ (a box, not a prism)</li>
<li><code>cube_Sum</code> — a loop that stopped too early</li>
<li><code>tuple_to_dict</code> — an index that ran off the end</li>
<li><code>max_product_tuple</code> — returned the pair, not the product</li>
</ul>
</div>
</div>

</div>

---
title: "The honest curve"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">Reading the result honestly</div>

## The learning curve: fast climb, then plateau.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/training-in-the-cloud-1.png" style="width:100%;" />
<p class="fig-caption">Held-out solve rate climbs early, then settles — the shape you hope for.</p>
</div>
<div>
<p class="text-sm">On held-out tasks the 0.5B model rose from <strong>44% to ~55–61%</strong>, then plateaued. Framed honestly: <strong>of the tasks it was failing, RL taught it to solve 14.</strong></p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">A real but <em>modest</em> jump — and saying so plainly is the point. We do not inflate results.</p>
</div>
</div>
</div>

</div>

---
title: "Running it for real"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Training real models is a systems problem</div>

## Real GPUs, and keeping the job alive.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/training-in-the-cloud-2.png" style="width:100%;" />
<p class="fig-caption">Deploy the job to the cloud so it survives your laptop closing.</p>
</div>
<div>
<p class="text-sm">Project two ran on <strong>Modal</strong> — rented cloud GPUs (H100s). A hard-won lesson: a naive background job dies when your laptop session ends. The fix is to <strong>deploy it fully server-side</strong> so it runs independent of you.</p>
<div class="callout-quiet mt-2">
<h4 style="color: var(--accent-2);">Everything is public</h4>
<p class="text-sm" style="margin:0;">Full <strong>paper</strong>, <strong>code</strong>, and two sites: a beginner before/after site and a deeper research site.</p>
</div>
</div>
</div>

</div>

---
title: "What RL holds on the GPU"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">The systems side — how many GPUs, and for what</div>

## What reinforcement learning must hold on the GPU.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/gpu-memory-picture.png" style="width:100%;" />
<p class="fig-caption">The model, its gradients, the optimizer state — and the reward, which needs no GPU at all.</p>
</div>
<div>
<p class="text-sm">RL is memory-hungry because one GPU must hold several things at once: the <span class="v-act">policy</span> (the model weights), its <strong>gradients</strong>, and the <strong>optimizer state</strong> (Adam roughly doubles the model's footprint) — plus the <strong>generation</strong> cache that produces the rollouts.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Two happy simplifications here: <strong>no reference model</strong> (our GRPO uses none), and the <span class="v-reward">reward is a program</span> — running the tests on the CPU — <strong>not</strong> a reward model on the GPU.</p>
</div>
</div>
</div>

</div>

---
title: "One H100, both jobs"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">Our allocation</div>

## One H100 per run — generating and training on the same card.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/gpu-one-h100-lora.png" style="width:100%;" />
<p class="fig-caption">The same model generates the attempts and takes the gradient step — colocated.</p>
</div>
<div>
<p class="text-sm">The simplest possible allocation: <strong>one H100</strong>, with the <em>same</em> model doing both the generation (rollouts) and the gradient update. Nothing is split across GPUs.</p>
<div class="callout-quiet mt-2">
<h4 style="color: var(--accent-2);">LoRA makes even 7B fit</h4>
<p class="text-sm" style="margin:0;">We freeze the base weights and train only tiny <strong>LoRA adapters</strong>, so the gradients and optimizer state stay small — a 7B model trains comfortably on a single 80&nbsp;GB card.</p>
</div>
</div>
</div>

</div>

---
title: "Many runs, many GPUs"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Scaling by fan-out, not by sharding</div>

## Sweeps are many one-GPU jobs, run in parallel.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/gpu-fanout.png" style="width:100%;" />
<p class="fig-caption">Each model size, seed, and reward variant is its own independent H100 job.</p>
</div>
<div>
<p class="text-sm">To compare 0.5B, 1.5B and 7B, several seeds, and the plain-vs-un-cheatable reward, we didn't build one giant job — we <strong>fanned out</strong> many <em>independent</em> single-GPU runs across H100s, each deployed fully server-side on Modal.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Because the runs never talk to each other, this needs <strong>no distributed coordination</strong> — and, as we'll see, no specialised RL framework either. <strong>Project 3 will change that.</strong></p>
</div>
</div>
</div>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">A completed experiment — and your launch point</div>

# Can the model<br/>*cheat* the tests?

<p class="lede mt-5">Because the reward is "did the visible tests pass?", a clever enough model can learn to satisfy those <em>specific</em> checks without truly solving the problem — <strong>reward hacking</strong>. We built a reward to resist it and ran the experiment. The next three slides are <strong>what we found</strong> — final results — and then it is yours to extend.</p>

</div>

---
title: "It catches the cheats"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">Result 1 · deliberate cheats</div>

## An un-cheatable reward catches what the plain one misses.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/iso-catches-cheats.png" style="width:100%;" />
<p class="fig-caption">Against hard-coded / memorised answers: the plain reward catches 0%, ours catches 93%.</p>
</div>
<div>
<p class="text-sm">We built a reward <em>designed to resist gaming</em> — the <strong>isomorphic perturbation reward</strong> (the how is in the paper; here we care only about what it did).</p>
<p class="text-sm mt-2">Against deliberate cheats — hard-coded outputs, lookup tables — it caught <strong>6 of 6</strong> live in real sandboxes, and <strong>93%</strong> across 12,000 samples. The plain "did the visible tests pass?" reward caught <strong>0%</strong>.</p>
</div>
</div>

</div>

---
title: "The honest tie"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Result 2 · the honest negative</div>

## On *natural* mistakes, it ties the plain reward.

<div class="grid grid-cols-2 gap-8 mt-3">
<div class="callout-quiet">
<h4 style="color: var(--accent-1);">What we hoped</h4>
<p class="text-sm" style="margin:0;">That the un-cheatable reward would also catch ordinary, honest bugs — the kind a model writes by accident.</p>
</div>
<div class="callout-quiet">
<h4 style="color: var(--accent-2);">What we measured</h4>
<p class="text-sm" style="margin:0;">A near-tie: precision <strong>0.79</strong> (plain) vs <strong>0.78</strong> (ours). It kills memorisation, but naturally-wrong code needs genuinely new test inputs to expose.</p>
</div>
</div>

<p class="text-sm mt-3">This is the honest, mature result — and a good research lesson: <strong>report the negative</strong>. (One tweak did lift its recall from <strong>0.90 to 0.97</strong> without hurting anything else.)</p>

</div>

---
title: "The RL payoff"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">Result 3 · the payoff, in a matched RL run</div>

## Swap only the reward — and true correctness rises.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/iso-rl-payoff.png" style="width:100%;" />
<p class="fig-caption">Plain reward: the visible score climbs, hidden correctness stays flat, the gap grows.</p>
</div>
<div>
<p class="text-sm">The decisive test: two identical training runs, changing <em>only</em> the reward. With the plain reward, the visible score climbed to <strong>0.83</strong> while true (hidden-test) correctness stayed <strong>flat at 0.63</strong> — the reward-hacking gap grew to <strong>0.20</strong>.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">With the un-cheatable reward, the gap stayed small and hidden correctness actually <strong>rose to 0.67</strong> — the only arm that got genuinely more correct.</p>
</div>
</div>
</div>

</div>

---
title: "Isomorphic homework"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-5);">Your homework</div>

## A real research project, ready to run.

<div class="grid grid-cols-2 gap-8 mt-3">
<div>
<p class="text-sm">That is the whole result: a reward that <strong>kills memorisation</strong>, <strong>ties on natural bugs</strong> (an honest negative), and, in a matched RL run, makes the model <strong>genuinely more correct</strong>.</p>
<p class="text-sm mt-2">It is a complete, publishable research thread — paper plus runnable code — and it is yours to extend.</p>
</div>
<div class="callout">
<h4 style="color: var(--accent-5);">Get started</h4>
<p class="text-sm" style="margin:.2rem 0 0;">Reproduce the "magic-moment" demo (watch a cheat get caught in a sandbox), then try your own perturbations and reward variants.</p>
<p class="text-sm mt-2" style="margin:.4rem 0 0; color: var(--ink-3);">→ the swe-rl-ipr paper &amp; code (linked on the project-two site)</p>
</div>
</div>

</div>

---
title: "Infrastructure grows with the task"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">A natural progression — why the next project needs more</div>

## The infrastructure grows with the task.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/infra-grows-with-task.png" style="width:100%;" />
<p class="fig-caption">Single-turn on one GPU needs no framework; long agentic rollouts do.</p>
</div>
<div>
<p class="text-sm">Project 2 was <strong>single-turn and short</strong>: one GPU, generation and training colocated, a ~200-line from-scratch GRPO loop. It needed <strong>no external RL framework</strong> — we controlled every line.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Project 3 is <strong>long, multi-turn, and agentic</strong>: many parallel sandbox rollouts, an async inference engine that mustn't idle, a distributed actor and learner. <em>That</em> is when you reach for a real framework — <strong>SkyRL</strong> and <strong>Prime Intellect's prime-rl</strong>, both of which we are actively using.</p>
</div>
</div>
</div>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 4 · Project three</div>

# A world model,<br/>for free. <span style="color: var(--accent-5);">ECHO.</span>

<p class="lede mt-5">Remember the masked observation tokens from Project 2 — the ones RL <em>throws away</em>? ECHO stops ignoring them. By also learning to <strong>predict</strong> what the terminal says back, a terminal agent builds a <em>world model</em> of its computer — for almost no extra cost. This is agentic RL's next step, and its most surprising one.</p>

</div>

---
title: "The signal RL throws away"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">Picking up exactly where agentic RL left off</div>

## Every rollout is full of evidence we're ignoring.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/echo-thrown-away.png" style="width:100%;" />
<p class="fig-caption">The reward is one sparse number; the observations are a dense, rich stream — normally discarded.</p>
</div>
<div>
<p class="text-sm">Recall the two kinds of tokens in a trajectory. The <span class="v-act">actions</span> get a single sparse reward at the very end. The <strong>observations</strong> — stdout, errors, files, logs — are rich and dense, and standard RL <strong>trains on none of them</strong>.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Even a <em>failed</em> rollout, which earns no reward, is full of evidence about how the computer behaves. ECHO's whole idea is to stop wasting it.</p>
</div>
</div>
</div>

</div>

---
title: "What is a world model"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">The idea from everyday life</div>

## A world model is a hunch about what happens next.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/what-is-a-world-model-1.png" style="width:100%;" />
<p class="fig-caption">Before you flip the switch, you already expect the light. That's a world model.</p>
</div>
<div>
<p class="text-sm">A <strong>world model</strong> is the ability to <em>predict what happens next</em>. For a terminal agent, that means: given the commands so far, predict what the terminal will print back.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">And notice — predicting the observations <em>is</em> learning to model the environment. The signal we were throwing away is the world model, waiting to be learned.</p>
</div>
</div>
</div>

</div>

---
title: "ECHO un-masks the observations"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-5);">The move — the payoff of the token mask</div>

## ECHO un-masks the observations and predicts them.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/echo-unmask.png" style="width:100%;" />
<p class="fig-caption">The observation tokens that Project 2 masked out are now a prediction target.</p>
</div>
<div>
<p class="text-sm">In Project 2, the observation tokens were <strong>masked</strong> — the model never trained on them. ECHO flips that: it keeps training the actions with GRPO, and <strong>additionally trains the model to predict the observation tokens</strong>.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Nothing new is collected. The observations were already in every rollout — ECHO simply stops throwing them away.</p>
</div>
</div>
</div>

</div>

---
title: "One forward pass, two jobs"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-5);">The loss — for free</div>

## Two jobs from the same forward pass.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/echo-one-extra-loss-2.png" style="width:100%;" />
<p class="fig-caption">Act (GRPO) and predict-the-reply share one forward pass — no extra rollouts.</p>
</div>
<div>
<p class="text-sm">The recipe is one line: keep GRPO on the actions, and add a small term that trains the model to predict the environment's tokens.</p>
<div class="equation-small mt-1">
L<sub>ECHO</sub> = L<sub>GRPO</sub>(<span class="v-act">actions</span>) + <span class="v-gamma">λ</span> · L<sub>env</sub>(<span class="v-value">observations</span>)
</div>
<p class="text-sm mt-1">With a small <span class="v-gamma">λ ≈ 0.05</span>. Because <code>L_env</code> reuses the <em>same forward pass</em>, it needs <strong>no extra rollouts and no extra model</strong> — a world model, essentially for free.</p>
</div>
</div>

</div>

---
title: "Does it learn a world model"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">Does it actually learn one? Yes.</div>

## It really does learn how the terminal behaves.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/echo-world-model-quality.png" style="width:100%;" />
<p class="fig-caption">ECHO sharply reduces its surprise at the terminal's replies; plain GRPO barely moves.</p>
</div>
<div>
<p class="text-sm">The clearest evidence: measure how <em>surprised</em> the model is by the terminal's output. Under ECHO, that surprise <strong>drops sharply</strong> — even on held-out trajectories it never generated. Under plain GRPO, it <strong>barely changes</strong>.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">So the world model is real, not incidental: ECHO learns to anticipate the computer, and that anticipation transfers to new tasks.</p>
</div>
</div>
</div>

</div>

---
title: "What it buys you"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-4);">Three wins from one extra loss</div>

## A dense signal changes what RL can do.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/echo-what-you-get.png" style="width:100%;" />
<p class="fig-caption">A real world model, expert-level results without demos, and self-improvement without a verifier.</p>
</div>
<div>
<p class="text-sm">The paper reports three striking wins. ECHO's policy <strong>predicts the terminal better</strong> (even on new tasks); it <strong>matches expert-SFT-then-GRPO without any expert demonstrations</strong>; and the observation loss <em>alone</em> enables <strong>verifier-free self-improvement</strong> on out-of-distribution tasks.</p>
<div class="callout-quiet mt-2">
<p class="text-sm" style="margin:0;">Turning a sparse-reward problem into a <em>dense</em> one is what makes all three possible.</p>
</div>
</div>
</div>

</div>

---
title: "Tested in a real terminal"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">The benchmark</div>

## Measured on 89 real terminal tasks.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agents-in-a-real-terminal-1.png" style="width:100%;" />
<p class="fig-caption">Real tasks — fix a bug, configure a tool, make a test pass — solved by typing commands.</p>
</div>
<div>
<p class="text-sm">Agents are evaluated on <strong>TerminalBench-2.0</strong> — <strong>89 genuine terminal tasks</strong> — with the base models <strong>Qwen3-8B</strong> and <strong>Qwen3-14B</strong> (open-weight, so it's reproducible). Rollouts run in real, stateful cloud sandboxes.</p>
<div class="callout-quiet mt-2">
<p class="text-sm" style="margin:0;">The authors call the terminal "the closest thing a language model has to an <em>embodied</em> setting" — a body that acts and perceives.</p>
</div>
</div>
</div>

</div>

---
title: "The result"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">The result — and how to read it</div>

## Adding the one extra loss roughly *doubles* the score.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/agents-in-a-real-terminal-4.png" style="width:100%;" />
<p class="fig-caption">Same everything — only the extra loss added — and pass@1 roughly doubles.</p>
</div>
<div>
<p class="text-sm">In a controlled A/B, ECHO roughly <strong>doubles</strong> GRPO's pass@1: <strong>8B: 2.70 → 5.17</strong>; <strong>14B: 5.17 → 10.79</strong>.</p>
<div class="callout mt-2">
<p class="text-sm" style="margin:0;">Read it honestly: these are a handful of 89 tasks and depend on private training data, so the <strong>relative doubling</strong> is the claim — not the exact number.</p>
</div>
</div>
</div>

</div>

---
title: "Our replication"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">Reproducing it — real ML is messy</div>

## Matching the *system* to the shape of the task.

<div class="grid grid-cols-2 gap-8 mt-2">
<div>
<p class="text-sm">Reproducing ECHO is where the earlier progression bites. Our first trainer (<strong>SkyRL</strong>) was <strong>synchronous</strong>: the inference engine sat idle ~15 minutes during each long rollout, and a watchdog killed it as "crashed" — every config failed the same way.</p>
<p class="text-sm mt-2">The fix is a <strong>fully asynchronous</strong> trainer (<strong>prime-rl</strong>), where the engines never idle — exactly the framework the task's shape demands.</p>
</div>
<div class="callout">
<h4 style="color: var(--accent-2);">Where we are right now</h4>
<p class="text-sm" style="margin:.2rem 0 0;">The ECHO extra-loss is implemented and <strong>numerically verified active</strong> — we can watch the environment-token loss fire — and the small smoke test passed. The full 8B / 14B runs are <strong>training as this lecture goes out</strong>; we'll drop in our measured curves the moment they land.</p>
</div>
</div>

<p class="text-sm mt-3" style="color: var(--ink-3);">The moral, one more time: matching your training <em>system</em> to the <em>shape</em> of the task matters as much as the algorithm.</p>

</div>

---
layout: center
---

<div class="section-cover max-w-4xl">

<div class="eyebrow mb-3">§ 5 · The big picture</div>

# The environment<br/>is the teacher.

<p class="lede mt-5">Three projects — a laptop, the cloud, a live terminal — and underneath, the same move every time. Let us name the thread that runs through all of it.</p>

</div>

---
title: "One idea, three projects"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-2);">The through-line</div>

## Nobody labels the answer. The environment grades the tries.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/the-teacher-is-the-environment-1.png" style="width:100%;" />
<p class="fig-caption">The same RL loop, with a different environment plugged in each time.</p>
</div>
<div>
<table class="text-sm">
<thead><tr><th>Project</th><th>Environment</th><th>Result</th></tr></thead>
<tbody>
<tr><td>Mini-SWE-RL</td><td>tests (1/0)</td><td>66.7 → 73.3%</td></tr>
<tr><td>Agentic RL</td><td>MBPP+ tests</td><td>44 → 51, 14 fixed</td></tr>
<tr><td>ECHO</td><td>a real terminal</td><td>~2× vs GRPO</td></tr>
</tbody>
</table>
<p class="text-sm mt-2">Different worlds, one recipe: a checkable goal, trial and error at scale, and reinforcement toward what worked.</p>
</div>
</div>

</div>

---
title: "Where this goes next"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-3);">What's still hard</div>

## Longer tasks, whole repos, world models everywhere.

<div class="grid grid-cols-2 gap-8 mt-2 items-center">
<div>
<img class="figimg" src="/figures/where-this-goes-next-1.png" style="width:100%;" />
<p class="fig-caption">The honest open problems — and the exciting ones.</p>
</div>
<div>
<ul class="text-sm">
<li><strong>Un-cheatable rewards</strong> — your isomorphic homework.</li>
<li><strong>Longer, multi-step tasks</strong> — real work is many steps, not one function.</li>
<li><strong>Whole real repositories</strong> — thousands of files in Docker (SWE-bench), not toy puzzles.</li>
<li><strong>World models everywhere</strong> — an agent that models the whole codebase, not just the terminal (back to ECHO).</li>
</ul>
</div>
</div>

</div>

---
title: "Build it yourself"
---

<div class="text-left max-w-5xl">

<div class="eyebrow mb-2" style="color: var(--accent-1);">You now know enough to build this</div>

## Everything is yours to run.

<div class="grid grid-cols-2 gap-8 mt-3">
<div class="callout-quiet">
<h4 style="color: var(--accent-2);">Read &amp; learn</h4>
<p class="text-sm" style="margin:.2rem 0 0;">The companion book <strong>"Teaching Machines to Code"</strong> — the whole stack, from scratch, heavily illustrated. Every figure in this deck comes from it.</p>
</div>
<div class="callout-quiet">
<h4 style="color: var(--accent-1);">Run the projects</h4>
<p class="text-sm" style="margin:.2rem 0 0;"><strong>Mini-SWE-RL</strong> (train on your laptop), the <strong>agentic-RL paper + code</strong>, and the <strong>isomorphic homework</strong> — all public.</p>
</div>
</div>

<div class="callout mt-4">
<p class="text-sm" style="margin:0;"><strong>The one thing to remember:</strong> teaching a machine to code is not magic and not locked in a big lab. Give it a checkable goal, let it try, let the environment grade it — and reinforcement learning does the rest. <em>The tests are the teacher.</em></p>
</div>

<hr class="rule-short mt-5" />
<p class="text-sm mt-3" style="color: var(--ink-3);">Vizuara AI Labs · RL in Production · thank you.</p>

</div>
