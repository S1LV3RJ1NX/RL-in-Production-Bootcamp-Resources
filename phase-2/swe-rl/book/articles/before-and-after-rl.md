The most convincing proof that reinforcement learning taught a model anything is embarrassingly simple: take the *same* model, hand it the *same* problem, once before training and once after, and watch it fail the first time and succeed the second. No aggregate percentages to squint at, no cleverness — just a broken line of code turning into a correct one. In this chapter we do exactly that, with five real bugs a small model could not fix on Monday and could fix by the time training finished.

Everything here comes from **project two**, the step up from the toy puzzles of [project one](training-on-a-laptop.html) to real programming tasks, run for real on rented cloud computers. The tasks come from a public collection called **MBPP+** (a set of **378** small, real Python problems, each with a plain-English request like *"write a function to…"*). And the star of the chapter is deliberately the *smallest* model we tried.

## Why we feature the tiny model

We trained three sizes of the same model family — **0.5B**, **1.5B**, and **7B** (that "B" is *billions of parameters* — the tunable numbers inside the model; more of them means a bigger, more capable, more expensive brain). Your instinct might be to show off the biggest one. We do the opposite, and there is an honest reason.

The **7B** model already solved about **84%** of the tasks *before we trained it at all*. It was near the ceiling — there was almost nothing left to fix, so there was almost no learning to *show*. The **0.5B** model, by contrast, started low and clumsy. That gave it the most room to grow, and room to grow is the whole story we are trying to tell.

[[note: aha || Pick the subject that has somewhere to go. A student already scoring 98% cannot demonstrate that your teaching works — a student at 44% can. The most improvable model tells the clearest before/after story, even though it is the weakest model.]]

[[fig: A hand-drawn "headroom" figure titled "why the small model?". Two vertical bars in black. LEFT bar labeled "7B model": filled almost to the top with a green "84% solved BEFORE" mark near the ceiling, and a tiny orange bracket at the very top labeled "room to improve = almost none". RIGHT bar labeled "0.5B model": filled only a little, a green "44% solved BEFORE" mark low down, and a tall orange bracket above it labeled "room to improve = lots". Dashed takeaway box in black: "choose the learner with the most headroom". || The 7B model starts near the ceiling, so training can barely move it. The 0.5B model starts low — which is exactly why its learning is easy to see.]]

## The headline: 44 to 51

Here is the honest result. On **100** held-out tasks — problems set aside and never used for training, so the model could not have memorized them — the 0.5B model solved **44** before training and **51** after. [[sn: "Held-out" is the golden rule of measuring learning. If you test on the very problems you trained on, a model can score well by memory alone, which proves nothing. So you hide a fresh batch away and only ever test on those. 44 out of 100, then 51 out of 100, both measured on the hidden batch.]] A modest jump of seven — and we would rather tell it the useful way.

Of the tasks the model was *failing*, RL taught it to solve **14** of them. (It slipped backward on a few it had barely been getting, which is why the net gain is seven, not fourteen — training nudges the whole model, and a nudge that helps one problem can occasionally jostle another.) Fourteen genuine bugs cracked is the number to hold onto.

If you watch the training unfold step by step, the shape is telling. The fraction of tasks solved **rises sharply in the first steps — from about 0.44 up to roughly 0.55–0.61 — and then plateaus**, flattening out and holding. The model learns fast, then holds. This whole run took just **80** steps on one rented **H100** (a high-end data-center GPU — a specialized chip that makes training fast).

[[fig: A hand-drawn learning curve titled "80 steps on one GPU". Axes in black: x-axis "training step" from 0 to 80, y-axis "fraction of tasks solved" from 0.0 to 1.0. A wobbly green line starts at a red-dashed level labeled "0.44 (before)", climbs steeply over the first steps, and levels off at a green-dashed band labeled "~0.55-0.61 (after)". An orange note beside the steep part reads "learns fast". A green tag near the plateau reads "14 fails -> passes". Dashed takeaway box in black: "a real, modest climb — then it holds". || The honest curve: a quick early rise from 0.44, then a plateau. Real learning looks like this — a burst, then a level.]]

## What the reward actually was

Before the examples, one reminder of *how* the model learned, because it explains why the fixes are trustworthy. There was no human grading the code. Every task ships with a few **visible tests** — tiny automatic checks like `assert add(2, 2) == 4` that either hold (the check "passes") or blow up with an error (it "fails") — and the reward was simply: did the visible tests pass? `reward = 1.0` if they all passed, `0.0` if any failed. The tests were the teacher, exactly as in the earlier projects, using the same learning rule, [GRPO](grpo-learning-from-a-group.html).

But each task *also* carries a larger, separate set of **hidden tests** — the honest ground truth the model never sees and never trains on. Those hidden tests are how we check whether a fix is *real* or merely *test-shaped*: a lazy model might satisfy the two or three visible checks by accident without truly solving the problem. Every "after" example below was confirmed to pass the hidden tests too. They are genuine fixes, not lucky guesses aimed at the few checks the model could see.

[[note: confusion || A common worry: "if the model is rewarded for passing the visible tests, won't it just learn to game *those specific* tests?" It can — a model rewarded for passing tests can sometimes learn to satisfy the exact checks without really solving the task. The hidden tests are our lie detector. If a fix passes the visible tests *and* the hidden ones, it genuinely understood the problem. (A sharper defense against this gaming comes in the final chapter.)]]

## Five bugs it learned to fix

Now the payoff. Here are five real functions, each shown as the buggy line the model wrote *before* training and the corrected line it wrote *after* — with the plain-English reason.

**1. `find_Volume` — the model forgot the ½.** The task: the volume of a triangular prism. Before, the model wrote `return length*width*height`. That is the formula for a plain *box*. But a triangular prism is *half* a box — its cross-section is a triangle, and the area of a triangle carries a factor of one-half. After training the model wrote `0.5*base*height*width`. It re-learned the missing `0.5`. Here is that fix drawn out —

[[fig: A hand-drawn before/after code card pair titled "find_Volume: half a box". LEFT card, pale-red fill, "before" in black with a red cross and the purple line "return length*width*height". A small red sketch of a solid rectangular box labeled "this is a BOX". RIGHT card, pale-green fill, "after" in black with a green check and the purple line "0.5*base*height*width". A small green sketch of a triangular prism labeled "half a box" with an orange "x 1/2" note. A bold blue arrow labeled "RL" runs left to right between them. Dashed takeaway box in black: "it re-learned the missing 0.5". || The volume bug in one picture: the model was computing the volume of a box and forgetting that a prism is half of one. Training restored the ½.]]

**2. `cube_Sum` — a loop that quit too early.** The task: sum the cubes of the first *n* even numbers. Before, the model looped with `range(2, n+1, 2)` — start at 2, step by 2, stop before `n+1`. That is wrong: for `n=2` it only reaches 2 and misses 4, because the *n*-th even number is `2*n`, not `n`. After, it wrote `range(2, 2*n+1, 2)`, which correctly walks 2, 4, 6, … up to the *n*-th even number. [[sn: `range(start, stop, step)` is Python's built-in counter: begin at `start`, add `step` each time, and stop *before* reaching `stop`. Off-by-one and off-by-*n* mistakes with `range` are among the most common real bugs there are — even experienced programmers trip on them.]]

[[fig: Two hand-drawn code cards side by side titled "cube_Sum: the loop bug". LEFT card, pale-red fill, titled "before" in black, with a red cross and one handwritten purple monospace line "range(2, n+1, 2)". Below it a small red number line 2,4,6 with only "2" circled and a red note "n=2 stops at 2 — misses 4!". RIGHT card, pale-green fill, titled "after" in black, with a green check and the purple line "range(2, 2*n+1, 2)". Below it a green number line 2,4,6,8 with "2" and "4" both circled and a green note "first n even numbers". A bold blue arrow labeled "reinforcement learning" points from the left card to the right. || The same model, before and after training: it learned that the n-th even number is 2*n, not n, and fixed the loop's stopping point.]]

**3. `tuple_to_dict` — reading past the end.** The task: turn a flat sequence of values into key→value pairs. Before, the loop read `tup[i+1]` — the item one step *past* the current one — and on the final item there was no "next" item to read, so the program crashed with an **IndexError** (Python's way of shouting "you asked for an item that doesn't exist"). After, the model stepped through the sequence *two at a time*, pairing each key with the value right after it, and never ran off the end. It went from a crash to a clean walk. Drawn out —

[[fig: Two hand-drawn code cards side by side titled "tuple_to_dict: off the end". LEFT card, pale-red fill, "before" in black with a red cross and the purple line "tup[i+1]". Above it a red row of boxes for a tuple, with a red arrow pointing just past the last box into empty space, labeled in red "no item here!" and a red tag "IndexError". RIGHT card, pale-green fill, "after" in black with a green check and the purple line "step 2 at a time". The same green row of boxes, now with curved green arrows pairing box 1 with box 2, box 3 with box 4, each pair labeled "key -> value". A bold blue arrow labeled "reinforcement learning" points from the left card to the right. || Before, the loop reached one slot past the end of the tuple and crashed; after, it walked in pairs and stayed safely inside.]]

**4. `max_product_tuple` — the right work, the wrong answer returned.** The task: find the largest product you can make from a pair of numbers in a list. Before, the model did the hard part correctly — it found the winning *pair* — but then handed back the pair itself instead of what was asked for. After, it returned their actual **product** (the two numbers multiplied). A subtle, very human bug: it solved the problem and then answered a slightly different question. Training taught it to return the thing that was actually requested.

**5. `find_Average_Of_Cube` — dividing by the wrong number.** The task: the average of the cubes of the first *n* numbers. An average is a sum divided by *how many things you added up*. Before, the model divided by the wrong total. After, it divided the sum of the cubes by `n` — the count of items. A one-symbol fix that captures what "average" means.

## The lesson that saved us from lying

There is one more example — the honest kind, the sort a textbook usually hides. While curating these before/after pairs, **two** of the "fixes" we almost showed you turned out to be fake. The model's function had been *correct all along*. It had only appeared to "fail" because the model, when it wrote its answer, had also appended a few of its *own* buggy test lines — and *those* were what failed, not the real code.

We caught it by doing the one thing you can always fall back on: we **ran the code** and read what actually happened, then checked it against the hidden tests. The moral is worth more than any single bug fix.

[[note: production || In real machine-learning work, you never fully trust a number you did not verify by execution. A metric can be right for the wrong reason. Grown-up systems run every candidate fix inside a safe sandbox, execute the real tests, and believe only what actually runs green. "It looks fixed" is not evidence; "it passed when we ran it" is.]]

You can see all five of these before/after examples running live, in a beginner-friendly walk-through, at `rl-teaches-code.vercel.app`. Same functions, same buggy and fixed lines, the tests turning from red to green in front of you.

## Where this leaves us

Fourteen real fixes, five of which you have now seen line by line, from a tiny model trained for **80** short steps. Modest, honest, and — because every "after" passes tests the model never saw — genuinely *learned*, not merely fitted. The point was never a big number; it was the *shape* of the change: broken code in, working code out, taught by nothing but tests and reward.

But running even this small experiment meant renting a real GPU in a data center and keeping a training job alive for hours without a laptop lid slamming shut on it. Training real models, it turns out, is as much a plumbing problem as a math one — and that plumbing is where we go next.
