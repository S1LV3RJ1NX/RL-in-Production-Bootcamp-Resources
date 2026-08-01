In the last chapter we agreed on the job: fix a bug so a failing test passes. But agreeing on a job is not the same as being able to *do* it. To do it, you need hands — a way to reach into the code, run it, and see what actually happens. For a programmer, and for the AI agents in this book, those hands are a thing called the **terminal**. This chapter is about what the terminal is, why it matters more than almost anything else, and why handing one to an AI is the whole game.

Here is the idea in one line, which we will unpack slowly: **the terminal is a text window where you type commands to the computer and it types text back.** No pictures, no buttons — just words in, words out. It sounds primitive. It is, in fact, the most powerful tool a programmer owns.

## What is a terminal, really?

Most of us know computers through their pretty face: icons you click, windows you drag, buttons that do things. That face is called a **graphical interface** — you point and click, and the machine responds with pictures. It is friendly, and it is also a narrow doorway. You can only do the things someone drew a button for.

The terminal is the other door. A **terminal** (also called the **command line** or the **shell** — the same thing under three names) is a plain text window. You type an instruction, press Enter, and the computer runs it and prints its reply right below. Then it waits for your next instruction. That back-and-forth — you type a line, it answers with a line — is the entire experience.

[[note: metaphor || A graphical interface is a restaurant with a fixed menu: you point at the pictures you are allowed to order. A terminal is walking into the kitchen and talking to the cook directly. It is less comfortable, and you had better know the words — but there is *nothing you cannot ask for.* Everything a computer can do, it can do from the terminal, whether or not anyone built a button for it.]]

Let me draw one, so the shape is concrete before we go further.

[[fig: A hand-drawn dark rounded rectangle, a "terminal window", titled in black above it "what a terminal looks like". Inside, on a dark background, four handwritten monospace lines in purple, each starting with a "$" prompt: "$ ls", then below it in plain light ink the reply "cart.py   discount.py   test_discount.py", then "$ python test_discount.py", then below it a red line "FAILED: test_final_price". A blue arrow curving from the "$" lines down to the reply lines, labeled in blue "you type -> it answers". Orange note pointing at the "$" symbol: "the prompt: the computer waiting for you". Dashed takeaway box in black at the bottom: "words in, words out — that is the whole thing". || A real terminal: you type a command after the "$" prompt, press Enter, and the computer prints its reply on the next line. Here it lists files, then runs the tests — which fail.]]

That little "$" at the start of each line you type is called the **prompt** — the computer's way of saying "I'm ready, go ahead." Everything after it is a command you gave. Everything without it is the computer answering.

## A handful of real commands

You do not need to memorize the terminal to understand this book — you need to recognize five or six commands, the way you can follow a recipe without being a chef. Here are the ones that come up again and again.

`ls` means **list** — "show me the files in the folder I'm standing in." Type `ls`, press Enter, and it prints their names. It is how you look around. [[sn: The name is short for "list," and terminal commands are almost all this terse — `ls`, `cd`, `cat`, `rm`. They were invented decades ago on slow machines where every keystroke counted, and the habit stuck. Cryptic, yes, but you only need a handful.]]

`cd` means **change directory** — "walk me into a different folder." A computer's files live in folders inside folders (a "directory" is just the old word for a folder), and `cd projects` steps you into the one called `projects`. This is how you move around.

`cat` means **show me this file** — `cat discount.py` prints the entire contents of the file `discount.py` right there in the window, so you can read the code without opening any app.

And the two that matter most for us: `python test_discount.py` **runs the tests** — it hands the file to Python, the language, and Python executes every check inside it and reports which passed and which failed. That single command is how the "did it work?" question from the last chapter gets answered, mechanically, in a fraction of a second.

Last, `git` is the command for **version control** — a system that tracks every change to the code over time, so you can see what was altered, undo a mistake, or bundle a fix up to propose it. When a real agent finishes a repair, `git` is how it saves and offers that change.

[[note: example || Watch a tiny session end to end. You type `ls` and see `discount.py  test_discount.py`. You type `cat discount.py` and read the buggy line. You type `python test_discount.py` and the computer prints `FAILED`. You edit the file, type `python test_discount.py` again — and this time it prints `PASSED`. Four commands, and you have found, understood, fixed, and *verified* a bug. That is software engineering, in miniature.]]

Drawn as a loop, that little session is the exact shape of everything to come.

[[fig: A hand-drawn cycle of three rounded cards connected by curved blue arrows, titled "the act-and-observe loop". Card 1 (black) "ACT": a small dark terminal strip showing a purple line "$ python test_discount.py". A blue arrow to Card 2 (black) "OBSERVE": a terminal strip showing the computer's reply, in red "FAILED: test_final_price". A blue arrow to Card 3 (black) "DECIDE": a small note reading "read the error -> edit the code", drawn with a tiny purple code card. A curved blue arrow loops from Card 3 back to Card 1 labeled "try again". Off to the side, a pale-green card showing a green "PASSED" with an orange note "loop ends when tests pass". Dashed takeaway box in black: "act -> observe the reply -> act again = an agent". || The terminal turns fixing a bug into a loop: type a command, read the reply, decide the next command, repeat until the tests pass. This act-and-observe cycle is exactly what reinforcement learning trains.]]

## Why this is the difference between talking and doing

Here is the point the whole book turns on. There are two very different things an AI can do when you show it a bug.

The first is to *talk about* the fix. You paste in some broken code, and the model says, in fluent English, "the problem is probably on line 4; you likely want to multiply instead of subtract." That is a description. It might be right. It might be confidently, completely wrong — and you would not know until you tried it yourself.

The second is to *actually make and check* the fix. The model runs `cat` to read the real file, edits the actual code, runs `python test_discount.py` to run the real tests, reads the real error the computer prints, and — if it is still failing — tries again. It does not *guess* whether its fix worked. It *finds out*, from the one authority that cannot be fooled: the tests.

[[fig: A hand-drawn two-panel comparison titled "describe vs do". LEFT panel, pale-red fill, labeled "no terminal": a robot icon with a speech bubble in red reading "you probably want to multiply here...", and a black label under it "only TALKS about the fix", with a red question mark "did it work? nobody knows". RIGHT panel, pale-green fill, labeled "with a terminal": the same robot icon wired by a blue arrow to a small dark terminal window running "$ python test_discount.py" with a green "PASSED" below it; black label under it "MAKES and CHECKS the fix". A bold orange arrow between the two panels labeled "give it a terminal". Dashed takeaway box spanning both: "a terminal turns advice into a verified result". || The whole difference in one picture. Without a terminal an AI can only describe a fix and hope. With one, it makes the change, runs the tests, and knows.]]

This is why we say a terminal is a coding agent's *hands*. A model with no terminal is a brilliant consultant locked in a room with a phone: it can tell you what to do, but it cannot touch anything, so it can never confirm it was right. A model *with* a terminal can reach out, do the work, and watch the result. That loop — act, observe the reply, act again — is exactly the trial-and-error loop that reinforcement learning is built to train. [[sn: Notice how neatly this connects to the last chapter. The reward in this book is "did the tests pass?" — a `1` or a `0`. The *only way to get that number* is to run the tests, and the only way to run the tests is through a terminal. The terminal is literally the wire that carries the reward back to the learner.]]

## Every project in this book acts through a terminal

You will meet this idea in all three projects, growing in directness each time.

[[fig: A hand-drawn horizontal "scale map" titled "the terminal gets more real, project to project". Three rounded boxes left to right connected by a blue arrow labeled "more real terminal ->". Box 1 (black): "PROJECT ONE — laptop puzzles" with a small sub-label "a mini env runs the tests for it" and a green tag "runs in ~30 min on a MacBook". Box 2 (black): "PROJECT TWO — real Python tasks" with sub-label "an environment runs the code + tests on cloud GPUs" and a green tag "trained on Modal / H100". Box 3, drawn as a dark terminal window (black title "PROJECT THREE — ECHO"): sub-label "the agent lives in a REAL terminal, typing commands over many steps" and a green tag "89 terminal tasks". Orange note under box 3: "types shell commands, reads the replies, step by step". Dashed takeaway box: "same hands, from a toy grip to the real thing". || The three projects, arranged by how literally the model touches a terminal. Project three, ECHO, lives entirely inside a real one.]]

In **Project One** and **Project Two**, the model does not type raw shell commands itself — instead a small program called an **environment** wraps the terminal for it. The environment takes the model's proposed fix, runs the tests behind the scenes (that `python test.py` step happens for it), and hands back the reward. It is a terminal with training wheels: the *doing* is real, but the interface is simplified so a small model can focus on the code. Project Two does this at real scale, running the code and tests on rented cloud machines called **Modal** — the same trial-and-verify loop, just on bigger hardware. [[sn: This wrapper is a real, standard idea, not a shortcut we invented. Big coding-agent systems use the same pattern: a sandboxed environment that runs the model's code safely and returns a score. Project One's environment mirrors one used on **8,100** real GitHub problems; ours is just small enough for a laptop.]]

**Project Three — ECHO — takes the training wheels off.** There, the agent lives inside a *real* terminal and solves each task the way a human developer would: by typing one shell command at a time, reading what the computer prints back, and deciding the next command based on that reply — over many steps, until the job is done. Its benchmark, **TerminalBench-2.0**, is exactly **89** real terminal tasks — fix a bug, configure a tool, make a test pass — every one of them done by typing commands. When we get there, you will see that giving a model a real terminal is not a convenience. It is what makes it an *agent* at all.

## Where this leaves us

A terminal is nothing fancier than a text window: you type commands, the computer types back. But that plain window is how anything actually gets *done* to code — how you look around with `ls`, read a file with `cat`, run the tests with `python test.py`, and see the real result instead of a hopeful guess. Handing that window to an AI is the difference between a model that describes a fix and one that makes it, checks it, and learns from the verdict.

We now have the hands. Next we need to understand the mind that guides them — how a language model reads a bug and writes a fix in the first place.
