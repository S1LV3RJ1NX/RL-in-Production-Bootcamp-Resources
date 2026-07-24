We have said, over and over, that the tests are the teacher and that a machine can learn to fix code by trial and reward. Now we build the thing that makes trial-and-reward possible. Before a model can *learn* to fix bugs, we have to hand it a **game**: a place where it can be given a broken function, take a swing at repairing it, and receive back a single honest number saying whether it won or lost. This chapter builds that game for our first real project — **Mini-SWE-RL** ([github.com/RajatDandekar/Mini-SWE-RL](https://github.com/RajatDandekar/Mini-SWE-RL)) — and it is small enough to run on a laptop.

## Why learning needs a game

Think about how a child learns to shoot a basketball. There is a **situation** (standing at the free-throw line, ball in hand), an **action** (they shoot), and a **score** (the ball goes in, or it doesn't). Repeat that a few hundred times and the aim improves — not because anyone wrote down the perfect elbow angle, but because each shot came back with a clear result. That triple — situation, action, score — is the whole engine of learning by trial and reward.

Reinforcement learning needs exactly the same three things. **Reinforcement learning** (learning by trying, getting a score, and adjusting to score higher next time) cannot even begin until you can hand the learner a situation, accept its action, and return a number. So our very first job is not to train anything. It is to build the basketball court.

[[note: metaphor || A game, in the RL sense, is just a court with a scoreboard. Situation in, action out, one number back. If you can wrap your problem in those three things — and code-fixing wraps beautifully — then a machine can practice it the same way a kid practices free throws.]]

## The environment: the world the agent acts in

In RL, that court has a name: the **environment** (the little world the learner acts inside — it holds the problem, receives the attempt, and returns the score). The learner is called the **agent**; here the agent is a language model, and the environment is a piece of code we write ourselves.

Our environment is called `CodeFixEnv`, and it is deliberately tiny. Its entire job is to hold a set of buggy functions, show one to the model, take back the model's proposed fix, and run the tests. That is it. Here is its whole interface — the handful of commands the rest of the program uses to talk to it:

```python
env = CodeFixEnv()
env.reset(puzzle_id)      # hand the model a buggy function
prompt = env.get_prompt() # format that bug as a question for the model
reward, done = env.step(fixed_code)  # run the tests -> 1.0 or 0.0
```

Here are those three verbs laid out as the game they build.

[[fig: A hand-drawn horizontal flow of three labeled panels joined by blue arrows, titled "three verbs = one game". Panel 1 (black) "reset(puzzle_id)": a hand pulling a card from a small deck labeled "puzzles", the drawn card a pale-red buggy code snippet in purple. Panel 2 (black) "get_prompt()": a speech bubble in blue reading "here is a function with a bug — please return a corrected version", with the buggy code tucked inside. Panel 3 (black) "step(fixed_code)": a dark terminal window running "$ run tests" with a green "reward = 1.0" and, below it, a red "reward = 0.0". A dashed takeaway box in black under all three: "pick a bug -> ask the model -> grade the fix". Orange note over Panel 3: "the only step that returns a number". || The full CodeFixEnv interface in three verbs: reset picks a bug, get_prompt asks the model, step grades the fix. Only step hands back a reward.]]

`reset` picks a puzzle and loads its broken code. [[sn: The name `reset` is borrowed from the standard "gym" convention that RL environments follow — you `reset` to start a fresh episode, then `step` through it. `CodeFixEnv` copies that convention exactly, which is why an RL practitioner can read it at a glance.]] `get_prompt` wraps that broken code in a plain instruction — *"here is a function with a bug, please return a corrected version"* — so the model knows what it is looking at. And `step` takes whatever code the model wrote back and does the one thing that matters.

## The heart of it: step runs the tests

Everything the game *means* lives inside `step`. When the model hands back a fix, `step` runs that fix against the puzzle's **tests** (small automatic checks that assert what the correct answer should be — like `assert is_balanced("()") == True`) and returns a **reward** (a single number saying how good the attempt was). The rule is as blunt as rules get:

> `reward = 1.0` if **all** the tests pass, else `reward = 0.0`.

No partial credit. No style points. This is a **binary** reward — one of exactly two values — and that severity is on purpose. The model does not get 0.7 for "almost right." Either the fix genuinely works on every check, or it does not count. The tests are the judge, the jury, and the entire scoreboard.

[[fig: A hand-drawn RL loop of three rounded cards joined by curved blue arrows, titled "CodeFixEnv, drawn out". Card 1 (black) "reset + get_prompt": a small buggy code card in purple reading "def is_balanced(s): ..." with a red squiggle under it, labeled "buggy function IN". Blue arrow to Card 2 (black) "the model writes a fix": a model icon with a fresh purple code card coming out, labeled "candidate fix". Blue arrow to Card 3 (black) "step: run the tests": a dark terminal window running "$ pytest" with two outcomes stacked — a pale-green row "all pass -> reward = 1.0" with a green check, and a pale-red row "any fail -> reward = 0.0" with a red cross. A curved blue arrow loops from Card 3 back to Card 1 labeled "next attempt". Dashed takeaway box in black: "buggy function in -> fix out -> tests give 1 or 0". || The whole environment as a loop: a bug goes in, the model writes a fix, the tests hand back a 1 or a 0. That single number is everything RL needs.]]

That is the entire game. A buggy function goes in, a fix comes out, the tests turn it into a `1` or a `0`. Loop that, thousands of times, and you have something a model can learn from.

## A real puzzle: the bracket-matcher

Abstract talk about "buggy functions" is easy to nod along to and hard to picture, so let us look at one real puzzle from the set. It is called `hard_balanced_parens`, and the task is to check whether the brackets in a string are properly matched — every opener has a matching closer, in the right order.

The catch is a subtle, very human mistake. The buggy version only knows about round brackets, `(` and `)`. It happily ignores square brackets `[]` and curly braces `{}` entirely — it just skips over them. So it will confidently declare the string `([)]` "balanced" when it plainly is not, and call `{[]}` balanced only by luck. The function *runs*. It never crashes. It simply gives the wrong answer on any input that uses the brackets it forgot about — exactly the quiet kind of bug we met in the opening chapter.

[[note: example || Feed `hard_balanced_parens` the string `"[]"`. A correct matcher returns `True`. The buggy one sees two characters it was never taught to handle, skips both, finds no unmatched `(`, and returns `True` as well — right answer, wrong reason. Now feed it `"[(])"`. The correct matcher returns `False`; the buggy one ignores every square bracket, sees a tidy `()`, and returns `True`. That second case is where the puzzle's test fails — and where the reward drops to `0.0`.]]

Here is that bug and its repair, side by side, in the shape we will keep coming back to.

[[fig: A hand-drawn before/after pair of code cards titled "hard_balanced_parens". LEFT card, pale-red fill, black title "before (buggy)", a red cross in the corner. Handwritten purple monospace lines: "for c in s:", "    if c == '(': stack.append(c)", "    if c == ')': stack.pop()   # ignores [] and {}". A red note with a dashed arrow to the last line: "only handles round brackets!". Below the card a red line "is_balanced('[(])') -> True  (WRONG)". RIGHT card, pale-green fill, black title "after (fixed)", a green check in the corner. Handwritten purple monospace: "pairs = {')':'(' , ']':'[' , '}':'{'}", "for c in s: ... check the matching opener". Below it a green line "is_balanced('[(])') -> False  (correct)". A bold blue arrow between the two cards labeled "the fix RL is trying to find". Dashed takeaway box in black: "same function — one quietly wrong, one right. tests tell them apart." || The bracket-matcher before and after. The buggy version silently ignores square and curly brackets; the fix handles all three. The test on '[(])' is what separates a reward of 0.0 from 1.0.]]

This is why the puzzles are good teaching material: each one is a real, recognizable Python trap. `hard_balanced_parens` is one of a set of fifteen hard puzzles built around algorithms and data structures; alongside it sit gems like `hard_scope_bug` (a Python quirk where a function written inside a loop accidentally shares one variable with the whole loop) and `hard_eval_rpn` (where Python's `//` division rounds toward negative infinity instead of toward zero). Every one is a bug a working programmer has written at three in the morning.

## Small enough for a laptop, real enough to matter

Here is the part that should feel almost too good. This exact game — an environment that hands out buggy code and grades fixes with pass-or-fail tests — is the *same shape* the professional systems use. The grown-up version is called **R2E-Gym**: a benchmark of **8,100** real bugs pulled from real GitHub projects, each one packaged inside **Docker** (a way to ship a program with its whole environment sealed in a box, so it runs identically anywhere). Docker lets each of those 8,100 bugs carry its own project and its own tests, safely isolated.

`CodeFixEnv` is that idea shrunk down until it fits on a desk. Same interface — `reset`, `get_prompt`, `step` — same binary "did the tests pass?" reward. Only the scale is different: instead of 8,100 dockerized GitHub repositories, a few dozen handwritten Python puzzles that a laptop can run in seconds.

[[fig: A hand-drawn scale map titled "same game, two sizes", two boxes side by side joined by a blue arrow labeled "same interface: reset / get_prompt / step". LEFT box (black outline, larger) "R2E-Gym": a stack of Docker-box icons in black with a green tag "8,100 real GitHub bugs" and a small note "needs a cloud of machines". RIGHT box (black outline, smaller, pale-green fill) "CodeFixEnv": a single laptop icon with a green tag "a few dozen puzzles" and a purple mini code card "hard_balanced_parens". A green label under the arrow: "binary reward: 1.0 if all tests pass, else 0.0". Dashed takeaway box in black: "identical shape — only the scale shrank". Orange note pointing at the laptop: "runs on YOUR machine". || The toy environment and the professional one are the same machine at different sizes. Shrinking the scale did not change the game — a bug goes in, tests hand back a 1 or 0.]]

And the model doing the fixing is small on purpose too. Mini-SWE-RL uses **Qwen2.5-Coder-1.5B** — a coding-focused language model with **1.5 billion parameters** (the tunable numbers inside a neural network; 1.5 billion of them here). By the standards of frontier models that is petite, which is exactly why it fits on an **Apple M4 Pro** laptop. During practice, the model generates its fixes through **Ollama** (a small program that runs a language model locally on your own machine), which answers each request in well under a second. No cloud, no data center, no bill.

[[sn: "Runs on a laptop" has one honest asterisk: generating the fixes happens through Ollama, which is fast and local, but the actual *learning* step uses a heavier library that needs to reach inside the model's parameters. Two tools, one model, one machine — but the learning half is the slower half.]]

## Where this is going

We now have the court and the scoreboard: `CodeFixEnv`, a game where a buggy function goes in, a fix comes out, and the tests return a clean `1.0` or `0.0`. What we do *not* yet have is a reason to believe every puzzle is worth training on. It turns out some puzzles are too easy for the model — it solves them every time — and some are too hard — it fails every time — and neither kind teaches it anything at all. Finding the puzzles that sit in the productive middle is the whole subject of the next chapter: the sweet spot.
