Every one of the three projects in this book — the laptop puzzle-solver, the cloud-trained code fixer, and the terminal agent with a mind for what comes next — learns by the same single rule. Understand this one rule and you understand the engine under all of them. It has a clumsy name, **GRPO** (Group Relative Policy Optimization), but the idea behind it is so plain you could explain it to a child at a spelling bee. This chapter is that explanation.

Here is the whole thing in one sentence, which we will spend the chapter unpacking: **give the model the same problem several times, let the tests score each attempt, then make the better-than-average attempts more likely and the worse-than-average ones less likely.** That is GRPO. Everything else is bookkeeping.

## Why try the same problem more than once?

Imagine a spelling bee where, instead of asking one child to spell "necessary," you ask eight children to spell it at once. Some get it right, some wrong. Now you know something you could never learn from a single child: you can see *which* answers were good, by comparing them to each other. That comparison is the seed of learning.

GRPO does exactly this. For one coding problem — say a buggy function that needs fixing — it asks the model to write a fix not once but several times over. This little batch of attempts at one problem is called a **group** (several fresh tries at the identical problem, so we can rank them against each other). A common size is **G = 8**: eight attempts at the same bug.

[[fig: A hand-drawn "group of tries" scene titled "one problem, eight tries". On the left, a single black box "PROBLEM: fix the buggy function". A blue arrow labeled "ask the model 8 times" fans out to eight small stacked cards, each labeled "try 1" through "try 8" in black. Numbered circle 1 sits on the problem box, circle 2 on the fan of arrows. Under each card a small placeholder tag reads "score = ?" in green. A dashed takeaway box on the right in black: "you can't rank 1 attempt against itself — you need a GROUP". || GRPO attacks one problem with a whole group of attempts, because learning here means comparing tries against each other.]]

To make the eight tries genuinely *different* — rather than the same answer copied eight times — GRPO turns up a dial called **temperature** (a knob for how much creative risk the model takes; low = play it safe, high = take chances). At temperature `0` the model always writes its single most-likely answer. Crank it to `0.8` and it takes small risks, so the eight fixes really do differ — some clever, some clumsy. [[sn: Nothing is magic about the number 8 or about `0.8`. Bigger groups give a cleaner ranking but cost more compute; a higher temperature gives more variety but also more nonsense. These are dials you tune. G = 8 at temperature 0.8 is just a sensible balance.]] That variety is the raw material the rest of the method feeds on.

## Scoring is the easy part: the tests do it

Once we have eight attempts, we need a score for each. This is where the quiet hero of the book steps in — **the tests are the teacher.** We do not read the eight fixes and judge by taste. We run each through its automatic tests (small pieces of code that check the fix works) and take the verdict they hand back: `reward = 1.0` if every test passes, `reward = 0.0` otherwise.

A **reward**, here, is simply a single number that says how good one attempt was — and in this book it is almost always just `1` or `0`, pass or fail. No opinions, no committee, no human in the loop. Eight attempts go in; eight `1`s and `0`s come out.

[[note: metaphor || Think of GRPO as a talent show with an automatic buzzer instead of celebrity judges. Eight contestants perform the same act; a machine buzzes `1` for the ones who nailed it and `0` for the rest. Nobody argues about who "felt" better — the buzzer already decided. Your only job now is to figure out what to do with those eight buzzes.]]

## The one clever idea: compare to the group's own average

Here is the heart of GRPO. Suppose our eight rewards for one bug come back as three passes and five fails. The natural thing to compute is the **group average** — the mean of the eight scores. Three passes out of eight is `3/8 = 0.375`.

Now comes the move that gives the algorithm its name. For each attempt we compute its **advantage** — how far above or below the group average it landed:

```
advantage_i = (reward_i − mean(rewards)) / std(rewards)
```

Read it slowly. `reward_i` is one attempt's score. `mean(rewards)` is the group average, `0.375`. Subtracting the average *centers* everything on the group: a passing attempt (reward `1.0`) sits above `0.375`, so it gets a **positive** advantage; a failing attempt (reward `0.0`) sits below, so it gets a **negative** one. The `/ std(rewards)` at the end divides by the group's **spread** (the standard deviation — roughly, how far apart the scores are), keeping the numbers on a comparable scale. [[sn: "Relative" in Group *Relative* Policy Optimization is exactly this: an attempt is never judged in absolute terms, only *relative to* its own groupmates. The same fix could count as good in a group of duds and bad in a group of stars. That is a feature, not a bug — it always compares like with like.]]

[[fig: A hand-drawn "group of tries" figure with a scoring line, titled "above or below the line?". Left: a black box "one bug". A blue arrow fans to eight cards labeled "try 1..8", each carrying a green "1" or red "0": reading 1,0,1,0,0,1,0,0 (three 1s, five 0s). Numbered circle 1 on the fan-out. To the right, a green handwritten line "group average = 3/8 = 0.375" and a long horizontal dashed black line drawn across the attempts at that height, with numbered circle 2 on the line. Orange up-arrows point at the three passing cards sitting ABOVE the dashed line; faint down-arrows at the five failing cards below it. Dashed takeaway box: "above the line = positive advantage; below = negative". || Eight scored attempts and their average line. GRPO does not ask "was this good?" but "did this beat the group average?"]]

## The nudge: more of the winners, less of the losers

We now have, for each attempt, a number saying whether it beat the group or lost to it. The final step turns those numbers into an actual change to the model — one formula:

```
loss = −Σ advantage_i × log P(attempt_i | prompt)
```

Do not panic at the symbols. `Σ` just means "add this up over all eight attempts." `log P(attempt_i | prompt)` is the model's own **confidence** that it would write that attempt again — how strongly it leans toward that answer. The whole formula is a machine for one instruction: **make the above-average attempts more likely, and the below-average ones less likely.** A positive advantage pushes an attempt *up* in the model's confidence; a negative one pushes it *down*. Do this across many problems and the model drifts, gently, toward whatever behavior passed the tests.

That is the full loop, drawn out — **try** eight times, **score** with the tests, **nudge** toward the winners, then come back around and try again:

[[fig: A hand-drawn cycle of four black-outlined rounded cards joined by curved blue arrows clockwise. Top: "MODEL — writes 8 fixes". Blue arrow to right card "TESTS — run each fix", with a green "reward = 1 or 0" tag hanging off it. Blue arrow down to bottom card "COMPARE — advantage vs group average", a small green "avg" line sketched inside. Blue arrow to left card "NUDGE — winners up, losers down", orange up/down arrows inside. A final blue arrow closes the loop from NUDGE back to MODEL. Centered orange label "try - score - nudge". Dashed black takeaway box below: "one turn of this loop = a hair of learning". || The whole of GRPO is this four-step loop spun over and over: try, score, compare, nudge.]]

A tiny by-hand trace makes it concrete. Say a group's eight rewards come back as `1,0,1,0,0,1,0,0`, so the average is `0.375`. Each *passing* try (reward `1`) gets a positive advantage `(1 − 0.375)/spread`, nudging the model to write answers like it more often; each *failing* try (reward `0`) gets a negative one, making those answers less likely. Three tries voted "up," five "down," and the model shifts a hair toward the three — and that single hair, repeated thousands of times, is training.

## Why GRPO is simpler than what came before

If you had studied reinforcement learning a few years ago, you would have met an older, heavier method called **PPO** (Proximal Policy Optimization). PPO needs a whole *second* neural network — a **critic**, sometimes called a value network — whose only job is to guess, in advance, how good each attempt is likely to be. That guess is the yardstick PPO measures attempts against. It works, but it means building and training a second model that can itself be wrong.

GRPO's quiet genius is to throw the critic away. It needs no network to guess how good an attempt "should" be, because it already has eight real attempts right there — and **the group's own average is the yardstick.** The other seven tries tell you whether the eighth was good. No second network, far less to build and debug.

[[note: confusion || A very common beginner worry: "Don't we need a separate model to judge each attempt?" That is precisely what PPO does with its critic — and precisely what GRPO skips. In GRPO the judge is not a model at all; it is arithmetic on the group. That is the single biggest reason GRPO is the friendly, from-scratch choice: fewer moving parts, one less thing to go wrong.]]

## The trap: a group that all agrees teaches nothing

There is one catch, worth burning into memory because it drives real decisions later in the book. Look once more at the advantage formula. What happens if all eight attempts get the *same* reward — all pass, or all fail?

Then every reward *equals* the average. Subtract the average and you get zero, every time. Every advantage is `0`, the loss contributes nothing, and the model learns nothing from that group. A group that all agrees — eight passes or eight fails — is a wasted group.

[[fig: A hand-drawn two-panel comparison titled "which groups actually teach?". LEFT panel, pale-red fill, labeled "flat group": eight cards all showing green "1" (all passed), a green line "average = 1.0" sitting exactly on top of them, a big black label "advantage = 0 for EVERYONE", and a red handwritten note "no learning signal". Numbered circle 1. RIGHT panel, pale-green fill, labeled "mixed group": eight cards showing a mix of green "1"s and red "0"s, a dashed average line through the middle, orange up-arrows above it and down-arrows below it, and a green note "clear signal!". Numbered circle 2. Dashed takeaway box spanning both panels: "GRPO needs DISAGREEMENT inside the group". || A group only teaches when it is mixed. All-pass and all-fail groups give zero learning signal, which is why problem difficulty must be tuned to the model.]]

This is the "sweet spot" you will meet again and again: GRPO learns only from problems where the model *sometimes* succeeds and *sometimes* fails. Too easy or too hard produces a flat, silent group. The interesting learning lives in the middle, where the group *disagrees with itself* — and every project in this book has to hunt for that zone.

## The same engine, three times over

What makes this worth a whole chapter is that we are not describing three tricks — we are describing *one*. This exact loop is the algorithm behind **DeepSWE**, an open coding agent that climbed from **42.2%** to roughly **59%** on a hard real-world benchmark of GitHub bugs. And it is used *identically*, only smaller, in all three of our projects: the laptop puzzle-solver ([Mini-SWE-RL](training-on-a-laptop.html)), the cloud-trained fixer of real Python tasks, and [ECHO](echo-a-world-model-for-free.html), the terminal agent that learns to predict its own environment. Same try-score-nudge. Same group average as the baseline. Same trap when the group agrees.

[[note: aha || If you remember one sentence from this book, make it this: **GRPO takes a group of scored tries, keeps the ones that beat the group's average, and drops the ones that didn't — and the tests do all the scoring.** Everything else in the three projects is scale, plumbing, and one clever twist on the reward. The heart is this paragraph.]]

We have the engine. Next we drop it into a real training loop on a laptop and watch a small model teach itself to fix bugs — try, score, nudge — from **66.7%** to **73.3%** in about half an hour.
