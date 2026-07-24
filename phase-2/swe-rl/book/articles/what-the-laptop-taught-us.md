In the last chapter we watched a small model teach itself, on a laptop, in about half an hour. It came out the other side a little better at fixing bugs — **66.7%** solved before, **73.3%** after. But a percentage is a cold thing. What did it actually *learn*? In this chapter we open the hood: we look at the real bugs it could not fix before training and could fix after, and then we ask the honest question — is a jump this small worth writing a book about?

The answer, it turns out, is yes, and for a reason that has almost nothing to do with the size of the number.

## Seven bugs it had never solved before

Here is the cleanest way to see what happened. Before training, there were a set of puzzles the model got wrong every single time. After thirty minutes of trial and reward, **seven of those puzzles it had never solved before** were now solved. Seven bugs went from red to green — not because someone showed the model the answers, but because the tests kept scoring its attempts and GRPO kept nudging it toward whatever passed.

[[fig: A hand-drawn "7 new bugs" before/after board titled "what the laptop taught it". LEFT column, pale-red fill, header "BEFORE — always failed": a stack of seven small red cards each with a red cross, four of them labeled in black: "scope_bug", "balanced_parens", "graph_cycle", "eval_rpn". RIGHT column, pale-green fill, header "AFTER — now solved": the same seven cards, now green with green checks. A single bold arrow in the middle, labeled in black "30 min of GRPO on a laptop", points left to right. A green tag above the right column: "+7 puzzles". Dashed takeaway box at the bottom in black: "nobody labeled the fixes — the tests did all the teaching". || Seven puzzles the model always failed before training, all passing after. The only teacher was the pass/fail signal from the tests.]]

The gain was not spread evenly, and that is worth saying plainly. On the **hard** puzzles — genuine algorithm problems — the model stayed at **73.3%**, unchanged; it was already near its ceiling there, and RL cannot conjure ability the model does not have. The whole improvement came from the **medium** puzzles, which climbed from **60.0%** to **73.3%** — a **+13.3**-point jump. Medium is where there was room to move, and so medium is where the learning showed up. [[sn: This is a recurring theme of the book: reinforcement learning sharpens what a model *almost* knows. It rarely teaches a genuinely new skill from nothing. If the model can never get a problem right even once, there is no winning attempt to reward, and nothing to learn from.]]

Let us actually look at some of these bugs. They are not made up for a textbook — they are real Python traps, the kind that catch human programmers too.

## A bug you can feel: the loop-variable trap

The puzzle called `hard_scope_bug` hides one of Python's most famous traps. Imagine you build a little list of functions inside a loop — say, one function for each number `0, 1, 2` — and each function is supposed to remember *its own* number. You would expect the first to return `0`, the second `1`, the third `2`. Instead, all three return `2`.

[[note: metaphor || Picture three people each told "remember the number on the whiteboard." They all glance at the board *later*, when you ask them — but by then you have wiped it and written the final number. All three say the same thing. They did not each copy the number down; they all pointed at the same board. That shared board is the loop variable, and pointing-at-it-later instead of copying-it-now is the whole bug.]]

In code, the trap is that a `lambda` (a tiny throwaway function) written inside the loop captures the loop variable **by reference**, not by value — it remembers *where* the number lives, not *what* it was at the time. By the time anyone calls those functions, the loop has finished and the variable holds its last value, so every function reports the same number. The fix is to make each function copy the value at creation time.

[[fig: A hand-drawn two-card "before/after" panel titled "hard_scope_bug: the loop-variable trap". LEFT card, pale-red fill, labeled "BEFORE" with a red cross: purple monospace "fns = [lambda: i for i in range(3)]" and below it three red output tags "2, 2, 2" with a red note "all point at the SAME i". RIGHT card, pale-green fill, labeled "AFTER" with a green check: purple monospace "fns = [lambda i=i: i for i in range(3)]" and below it three green output tags "0, 1, 2" with a green note "each snapshots its own i". A bold blue arrow between the cards labeled "learned to copy the value". Dashed takeaway box in black: "capture by VALUE, not by reference". || The famous closure trap, before and after. The buggy version shares one variable across all three functions; the fix hands each its own copy.]]

A human can stare at this for twenty minutes. The model, after training, gets it. [[sn: The everyday fix is to bind the value with a default argument, e.g. `lambda i=i: i`, which snapshots `i` at the moment the lambda is made instead of looking it up later. The model does not need to know *why* it works, only that this shape makes the test pass.]]

## Three more, in plain words

The other newly-solved bugs are just as concrete, and each one is a small, checkable idea:

- `hard_balanced_parens` asks whether the brackets in a string are properly matched. The buggy version only handled round parentheses `()` and silently ignored square `[]` and curly `{}` brackets — so it happily called `([)]` "balanced." The fix is to track all three kinds.
- `hard_graph_cycle` looks for a loop in a network of connected nodes. The buggy version could not tell the difference between a node it had **visited at some point** and a node that is **on the path it is currently walking** — and that difference is the entire point of cycle detection. You have only found a loop if you arrive back somewhere on your *current* trail, not just anywhere you have ever been.
- `hard_eval_rpn` evaluates arithmetic and trips over a quiet fact about Python: the `//` operator does not chop toward zero the way you might expect — it rounds *down*, toward negative infinity. So `-7 // 2` is `-4`, not `-3`. One wrong assumption about one operator, and the whole calculator gives wrong answers on negative numbers.

[[fig: A hand-drawn two-card "before/after" panel titled "hard_eval_rpn: one operator, wrong assumption". LEFT card, pale-red fill, labeled "BEFORE" with a red cross: purple monospace "-7 // 2  ->  expected -3" and a red note "assumes // truncates toward 0". RIGHT card, pale-green fill, labeled "AFTER" with a green check: purple monospace "-7 // 2  ==  -4 (rounds DOWN)" and a green note "handles negatives correctly". A bold blue arrow between them labeled "learned the real rule". Dashed takeaway box in black: "a real bug is often ONE wrong assumption about ONE line". || A single misunderstood operator, fixed. Most real bugs are exactly this local — one wrong belief about one line of code.]]

Notice what these four bugs have in common. Each is a single, precise misunderstanding — a bracket kind forgotten, a distinction not drawn, an operator misjudged. That is what real debugging usually *is*: not a heroic rewrite, but finding the one small place where a belief about the code was wrong. And crucially, for every one of them there was a **test** that said *pass* or *fail* with no argument. The model tried, the test judged, and over many rounds the model drifted toward the versions that passed.

[[note: aha || The model was never told *why* `//` rounds down or *what* a DFS path is. It was only ever told "this attempt passed" or "this attempt failed." Understanding, such as it is, emerged from thousands of tiny pass/fail verdicts. That is the strange power at the center of this whole book — competence grown from a single number.]]

## Is 6.7 points worth it?

Now the honest reckoning. **+6.7** points overall is a modest result. On a big public leaderboard it would not turn a single head. If this were a frontier lab claiming a breakthrough, you would be right to roll your eyes.

But that is not what this is, and comparing it to a frontier lab is exactly the wrong frame — because look at what a frontier system actually costs.

[[fig: A hand-drawn side-by-side "scale map" titled "same algorithm, wildly different scale". LEFT box, black outline, header "DeepSWE (the big lab)": a datacenter icon with green spec labels "Qwen3-32B", "64 H100 GPUs", "6 days", and a small "real GitHub bugs" tag. RIGHT box, black outline, header "Mini-SWE-RL (this project)": a single laptop icon with green spec labels "Qwen2.5-Coder-1.5B", "1 MacBook", "~30 min". Between the two boxes, a bold orange banner reading "SAME LOOP: GRPO" with a small purple tag "try -> score -> nudge". Below, a green note under the laptop "~20x smaller model". Dashed takeaway box spanning the width in black: "the recipe is identical — only the scale differs". || The gap between the frontier and your desk is a gap of scale, not of ideas. The GRPO loop on the laptop is the same one running in the datacenter.]]

**DeepSWE**, one of the strongest open coding agents, trains a **Qwen3-32B** model on **64 H100 GPUs** for **6 days** to score on real GitHub bugs. Mini-SWE-RL trained a **Qwen2.5-Coder-1.5B** model — roughly **twenty times smaller** — on **one MacBook** for **thirty minutes**. That is a difference of many orders of magnitude in model size, hardware, and time. [[sn: An H100 is a data-center GPU the size of a book that costs tens of thousands of dollars; DeepSWE used sixty-four of them at once. The laptop used exactly zero. The point is not that the laptop competes — it obviously does not — but that it runs *the same idea*.]]

And here is the thing that makes the small number matter: **the algorithm is identical.** Not similar, not inspired-by — the same [GRPO](grpo-learning-from-a-group.html) loop you read line by line in the last chapter is the loop DeepSWE runs. The datacenter adds scale and safety rails; it does not add a secret. When your laptop went from **66.7%** to **73.3%**, it did so by exactly the mechanism the big labs use.

[[note: production || Mini-SWE-RL is a complete, from-scratch miniature of the real pipeline, piece for piece. Its `CodeFixEnv` — the little gym that hands out buggy functions and returns `1.0` or `0.0` — plays the role that the industrial R2E-Gym (8,100 real GitHub problems in Docker) plays for the big systems. Its agent mirrors DeepSWE. Its trainer is GRPO, the same family that powers rLLM. Nothing is faked or hand-waved; it is the whole machine, shrunk until it fits on a desk.]]

## What the laptop really proved

So the seven bugs are the visible result, but they are not the real lesson. The real lesson is that reinforcement learning for software engineering is not a locked box that only opens for people with a warehouse of GPUs. Every essential gear — an environment that scores code with tests, a model that tries many times, an algorithm that rewards the winners — is small enough to hold in your hands and watch turn. The frontier and the laptop are separated by scale, and scale alone.

That is a liberating fact. It means you can *understand* this technology fully by building the tiny version, and everything you learn transfers straight up. It also means the honest, modest **+6.7** is not an embarrassment to explain away — it is proof that the whole loop works end to end, on hardware you already own, on real bugs with real fixes.

There is a catch, though, and the next project confronts it head-on. Our puzzles here were hand-written toys — clean, self-contained, one function each. Real programming tasks are messier, larger, and less forgiving. In the next chapter we leave the laptop's toy box behind and point the very same GRPO loop at real programming problems, run for real on cloud GPUs.
