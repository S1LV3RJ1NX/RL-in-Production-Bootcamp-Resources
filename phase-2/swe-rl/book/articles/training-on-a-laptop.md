We now have the two pieces we need. From the last chapters we have a **game** — a buggy function goes in, a fix comes out, and the tests hand back a `1` or a `0` — and we have **GRPO**, the rule for turning a pile of scored attempts into a slightly better model. This chapter snaps them together into the actual training loop that ran on a MacBook and took a small model from **66.7%** to **73.3%** in about **thirty minutes**. We will walk the whole loop once, slowly, with the real code.

The loop has just three steps, repeated over and over: **try, score, nudge.** Everything else is detail.

[[fig: A hand-drawn cycle of three big rounded cards connected by curved blue arrows, titled "the training loop". Card 1 (black) "TRY": a small model icon with the label "8 attempts at one bug" and a fan of tiny purple code cards. Card 2 (black) "SCORE": a dark terminal window running "$ run tests" with green "1" and red "0" tags scattered on the attempts. Card 3 (black) "NUDGE": the model icon again with a blue up-arrow "do more of the winners" and a faint down-arrow "less of the losers". A curved blue arrow loops from card 3 back to card 1 labeled "repeat ~10 times". Orange takeaway in a dashed box: "try -> score -> nudge, over and over". || The entire training loop in three steps. Every technical detail below hangs off one of these three cards.]]

## Step one: try (collecting rollouts)

To learn from trial and error, you first need trials. In RL these recorded attempts are called **rollouts**, and collecting them is step one.

For each buggy puzzle, we ask the model to write a fix — not once, but **eight times**. Why eight? Because GRPO learns by *comparison*: it needs several attempts at the *same* problem so it can see which ones did better than the others. One attempt tells you nothing about better-or-worse; eight attempts give you a spread to rank. [[sn: This group of attempts at one problem is the "group" in Group Relative Policy Optimization. In this project the group size is G = 8. Bigger groups give a cleaner ranking but cost more compute; 8 is a good balance on a laptop.]]

To get eight *different* answers instead of the same one eight times, we turn up a dial called **temperature**. At temperature 0 the model always writes its single most-likely answer; crank it to `0.8` and it takes small creative risks, so the eight attempts genuinely differ — some clever, some clumsy. That variety is the raw material RL feeds on.

```python
# one puzzle, eight attempts, at temperature 0.8
rollouts = []
for _ in range(8):
    fix = model.generate(prompt, temperature=0.8)  # a candidate fix
    reward, _ = env.step(fix)                       # run the tests: 1.0 or 0.0
    rollouts.append((fix, reward))
```

Across 15 training puzzles, eight attempts each, that is **120 rollouts**, collected in roughly **two minutes**. One practical trick made this fast: inference (generating the fixes) runs through **Ollama**, a lightweight local model server, which answers each request in well under a second — far quicker than the heavier machinery you need for the actual learning step. [[sn: The same model weights, `qwen2.5-coder:1.5b`, are used for both generating (via Ollama, which is fast) and for the gradient update (via HuggingFace Transformers, which exposes the model's internals). Ollama is for speed; HuggingFace is for learning.]]

## Step two: score (the tests hand back a number)

Scoring is the easy part, and that is exactly the point of this whole approach. We do not judge the fixes ourselves. We run each one through the environment's tests and take the reward it hands back: **`1.0` if every test passes, `0.0` otherwise.** For one puzzle, the eight rewards might come back looking like this:

[[fig: A hand-drawn "group of tries" figure. On the left, a single black box "one bug: hard_balanced_parens". A blue arrow fans out to eight small stacked cards labeled "attempt 1..8", each with a green "1" or red "0" on it: say attempts read 1,0,1,0,0,1,0,0 (three 1s, five 0s). To the right, a green handwritten line "group average = 3/8 = 0.375" with a horizontal dashed black line drawn across the attempts at that height. Orange up-arrows point at the three winning attempts (reward 1, above average); faint down-arrows at the five below. Dashed takeaway box: "above the line -> make more likely; below -> less likely". || Eight attempts at one bug, scored by the tests. The dashed line is the group average; GRPO simply pushes the model toward whatever sits above it.]]

Three of the eight attempts passed, five failed, so the **group average** is 3/8 = 0.375. That average is the crucial reference point. GRPO does not ask "was this attempt good in some absolute sense?" It asks "was this attempt **better or worse than its groupmates?**"

## Step three: nudge (the GRPO update)

Now the learning. For each attempt we compute its **advantage** — how far above or below the group average it landed:

```
advantage_i = (reward_i − mean(rewards)) / std(rewards)
```

Subtracting the mean centers the scores around the group; dividing by the spread (the **standard deviation**, `std`) just keeps the numbers on a sane scale. A passing attempt in our example lands *above* the 0.375 average, so it gets a **positive** advantage; a failing attempt lands below, so it gets a **negative** one. Then we adjust the model with a single rule:

```
loss = −Σ advantage_i × log P(attempt_i | prompt)
```

In words: **make the above-average attempts more likely, and the below-average ones less likely.** `log P(attempt | prompt)` is just the model's own confidence in having written that attempt; multiplying by a positive advantage and nudging to reduce the loss raises that confidence, and a negative advantage lowers it. Do this across all the puzzles and the model drifts, gently, toward the behavior that passed more tests.

[[note: confusion || A natural worry: "don't we need a second network — a 'critic' — to judge how good each attempt is?" That is what the older algorithm, PPO, does. GRPO's trick is to skip it entirely: the **group's own average is the judge**. The other seven attempts are the yardstick for the eighth. No critic network, far less to build — which is exactly why GRPO is the friendly choice for a from-scratch project.]]

Here is where the "sweet spot" from the previous chapter earns its keep. Look again at the advantage formula. If all eight attempts get the *same* reward — all pass, or all fail — then every reward equals the mean, every advantage is **zero**, and the update does nothing at all. A puzzle only teaches when its group is **mixed**: some 1s, some 0s. That is why we trained on the 18 "mixed" puzzles and set aside the ones the model always aced or always flunked. A flat group is a wasted group.

[[fig: A hand-drawn two-panel comparison titled "which groups actually teach?". LEFT panel, pale-red, labeled "flat group": eight cards all green "1" (or all red "0"), a green line "average = 1.0" sitting exactly on them, and a big black "advantage = 0 for everyone" with a red "no learning" note. RIGHT panel, pale-green, labeled "mixed group": eight cards reading a mix of 1s and 0s, a dashed average line through the middle, orange up/down arrows on either side, and a green "clear signal!" note. Dashed takeaway box spanning both: "RL needs disagreement inside the group". || Only a mixed group — some wins, some losses — produces a non-zero learning signal. An all-pass or all-fail group teaches nothing, which is the whole reason difficulty must be tuned.]]

## Putting the loop together

Stack the three steps and repeat them for about ten passes over the puzzles — ten **epochs** — and you have the entire trainer. Stripped to its skeleton, it really is this small:

```python
for epoch in range(10):
    for puzzle in mixed_puzzles:                 # only the puzzles that teach
        group = collect_rollouts(puzzle, n=8)    # TRY: 8 attempts
        rewards = [r for (_, r) in group]        # SCORE: tests -> 1/0
        advs = (rewards - mean(rewards)) / std(rewards)
        loss = -sum(a * logprob(fix) for (fix, _), a in zip(group, advs))
        loss.backward(); optimizer.step()        # NUDGE: one gradient step
```

That is GRPO. There is no hidden complexity waiting in the wings — the production systems add scale and safety rails, but this loop is the beating heart of all of them.

## The result

After roughly thirty minutes of this on an Apple M4 Pro laptop, the little Qwen2.5-Coder-1.5B model improved measurably. Its overall solve rate on the held-out puzzles rose from **66.7%** (20 of 30) to **73.3%** (22 of 30) — a real **+6.7** points. The gain was concentrated where there was room to grow: on the medium puzzles it jumped from **60.0%** to **73.3%**, a **+13.3**-point leap, while the hard puzzles (already near the model's ceiling) held steady. Most tellingly, the model **solved seven puzzles it had never solved before** — bugs we will look at, one by one, in the next chapter.

[[fig: A hand-drawn learning curve titled "30 minutes on a laptop". Axes in black: x-axis "training epoch" 0 to 10, y-axis "solve rate" 0% to 100%. A wobbly green line starts at a red-dashed level labeled "66.7% (before)" and climbs unevenly to a green-dashed level labeled "73.3% (after)", flattening near the end. A small orange note by the rise: "medium puzzles +13.3%". A green tag near the top-right: "7 new bugs solved". Dashed takeaway box: "same GRPO as the big labs — 1/1000th the scale". || The real training curve: a modest, honest climb from 66.7% to 73.3% in half an hour, with the biggest gains on the medium puzzles that had room to improve.]]

Modest? Yes — and honestly so. This is a 1.5-billion-parameter model on a laptop for half an hour, not a frontier system. But that is precisely what makes it remarkable.

[[note: production || The state-of-the-art system this mirrors, DeepSWE, trains a Qwen3-32B model on 64 H100 GPUs for six days to score on real GitHub bugs. Mini-SWE-RL uses a model roughly twenty times smaller, on one laptop, for thirty minutes. The scale differs by many orders of magnitude — but the algorithm on the page is **the same GRPO loop you just read.** That is the point: the idea is not locked inside a big lab. It fits on your desk.]]

We have watched a model teach itself to fix bugs, using nothing but its own attempts and the tests' verdicts. In the next chapter we look at *what* it learned — the seven specific bugs it cracked — and ask why that small result carries a much larger lesson.
