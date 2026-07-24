So far our machines have fixed *one* buggy function at a time: read the code, write a fix, get graded, done. But a real software job is rarely one move. You poke around, run a command, read what the computer says back, run another, and only then do you know what to fix. This chapter teaches a model to *live* in that back-and-forth — and shows a small, clever idea that gives it a kind of intuition almost for free. It is also the messiest project of the three, and the mess is the lesson.

## What a terminal agent actually does

Recall the **terminal** (the text-only window where you type commands and the computer types text back). A **terminal agent** is a model that solves a task not by writing one answer, but by holding a conversation with that window over many turns: type a command, read the reply, decide the next command, and repeat until the job is done.

[[fig: A hand-drawn dark terminal window titled "one task, many steps", rounded rectangle. Inside, alternating handwritten monospace lines: a purple "$ ls" then a black reply "app.py  test_app.py"; a purple "$ python test_app.py" then a red reply "FAILED: test_discount"; a purple "$ cat app.py" then a black reply "def final_price(...):"; a purple "$ edit app.py  (fix the line)"; a purple "$ python test_app.py" then a green reply "1 passed". Numbered circles 1 through 5 down the left edge marking the order. Orange note on the right with a curved arrow: "the model chooses each command AFTER seeing the last reply". Dashed takeaway box in black: "solving = a loop of command -> reply -> next command". || A terminal agent solves a task step by step: run a command, read what came back, decide the next one — exactly how a human developer works.]]

Notice the shape. The agent does not plan all five commands up front. After each command it *reads the reply* and lets that reply steer the next move. That is what makes it an agent, not a one-shot answer machine — and it is why training here is harder than anything before.

[[note: metaphor || Think of cooking in an unfamiliar kitchen. You do not write the whole plan before you start — you open a drawer, see what is in it, and *that* tells you your next move. The terminal is the drawer; the reply is what you find inside; the agent decides where to reach next from what it just saw.]]

## The benchmark: 89 real terminal tasks

To measure progress we need a **benchmark** (a fixed set of tasks everyone is scored on, so results can be compared honestly). This project uses **TerminalBench-2.0**, which is exactly **89** real terminal tasks — fix a bug, configure a tool, make a test pass, all by typing commands. The score is **pass@1**, the fraction of tasks the agent solves on its *first* try.

Keep that denominator in mind: 89 is a small number. Every task is worth a little over one percentage point — which matters a great deal when we talk honestly about the results.

The models taught are **Qwen3-8B** and **Qwen3-14B** — open-weight models (anyone can download the actual parameters), with 8 billion and 14 billion tunable numbers inside. Training uses the same [GRPO](grpo-learning-from-a-group.html) loop as the rest of the book: let the model try, score each attempt, nudge it toward the winners.

## The clever bit: a world model, for free

Here is the idea that makes this project special. When the agent types `ls`, it is about to read a reply — a list of files. A good developer already expects what that reply will look like *before it arrives*. That silent expectation is a **world model** (a model's built-in ability to predict what its environment will do next). Before you flip a light switch, you already expect the light to come on. That expectation is a world model in your head.

The project, called **ECHO** (a replication of a paper by that name), adds one small extra job to normal GRPO training. As usual, the model learns to *act* — to type good commands. But at the same time, reusing the very same **forward pass** (the single sweep of the input through the network that produces its output — the model's one bit of "thinking"), it also learns to *predict the terminal's reply*. Guessing what the computer will print teaches the model how the computer behaves — as a side effect of learning to act.

[[fig: A hand-drawn "extra loss wiring" scene titled "one forward pass, two jobs". Center: a black rounded box "the model reads the conversation so far". Two blue arrows leave it: top to a black box "ACT: type the next command" (green tag "the main GRPO job"); bottom to a black box "PREDICT: guess the terminal's reply" (green tag "the extra job", green label above "weight = 0.05"). A purple formula strip along the bottom: "loss = L_act + 0.05 x L_predict". Orange callout arrow at the shared box: "SAME pass -> almost no extra cost = 'for free'". Dashed takeaway box in black: "predicting the reply builds a world model on the side". || ECHO reuses one forward pass for two jobs: act, and predict the terminal's reply. The prediction job costs almost nothing and quietly teaches the model how the world behaves.]]

The whole extra idea is one line of math: `L_ECHO = L_GRPO + 0.05 · L_env`. The first term is the ordinary act-and-get-rewarded loss. The second, weighted by a small **λ = 0.05** (lambda, just the size of the knob), is a prediction loss on the terminal's replies — technically the model's *surprise* at what the terminal printed, which training pushes down. [[sn: "For free" is doing honest work here. It does not mean *zero* cost — it means *negligible*: the same forward pass, one extra loss term, so the compute barely changes. That is why the paper's title says the world model comes "for free."]] It reuses work the model was already doing, so it barely costs anything.

[[note: aha || The one sentence to keep: **an agent that learns to predict what the computer will say learns, for free, how the computer works** — and that quiet understanding makes it a better agent.]]

## Where the model actually runs: sandboxes

Every command-and-reply attempt has to run *somewhere*. You cannot let a half-trained model type arbitrary shell commands on your real laptop — one wrong `rm` could delete your files. So each attempt runs inside a **sandbox** (a safe, isolated container — a sealed-off mini-computer where untrusted commands run without touching the real machine).

This project runs its sandboxes on **Modal** (the cloud service that rents GPUs and runs your code on them). Each sandbox is a real, *stateful* terminal — it remembers: a file you create with one command is still there for the next, just like a real machine. That statefulness is essential, because the agent's whole method is to build on what earlier commands did.

## The war story: why the engine kept crashing

Now the mess — the part beginners rarely get to see, and the most useful part of the chapter: real machine learning is a systems fight, not just a math problem.

The team's first trainer was **SkyRL**. It worked — except it kept dying, the same way, on every configuration. The cause was a mismatch between *how SkyRL trains* and *the shape of the task*.

SkyRL is **synchronous** — it does one thing at a time, in lockstep. It would send the model off to run a long batch of terminal attempts (256 full episodes, then grading, then syncing the updated weights), which takes a while. Meanwhile the **inference engine** — the fast component whose only job is to generate the model's commands — had nothing to do. It sat idle for about **15 minutes** every round, waiting its turn.

[[fig: A hand-drawn timeline titled "sync vs async: why the engine crashed", two tracks. TOP track, pale-red, "SkyRL (synchronous)": a long red bar "rollouts: 256 episodes + grading + weight sync (~15 min)", and below it a red bar "inference engine: IDLE, waiting". A red clock at ~10 min, purple label "watchdog (~600s) fires", red skull "engine killed — assumed crashed", red cross "every config failed the same way". BOTTOM track, pale-green, "prime-rl (asynchronous)": two green bars overlapping in time — "rollouts running" and "inference engine: BUSY generating" — no idle gap, green check "no idle -> no crash". Orange note between tracks: "same task, different SYSTEM shape". Dashed takeaway box in black: "match the training system to the shape of the task". || The bug was not in the math. A synchronous trainer left the inference engine idle for ~15 minutes; a safety timer assumed it had crashed and killed it. An asynchronous trainer keeps the engine busy, so the crash disappears.]]

Here is the trap. Long-running programs often carry a **watchdog timer** — a safety timeout that assumes "if this component hasn't responded in a while, it must have crashed, so restart it." SkyRL's watchdog was set to about **600 seconds** (ten minutes). The idle inference engine, sitting quietly and correctly waiting for its turn, blew past ten minutes of silence — so the watchdog concluded it had died and killed it. The engine was perfectly healthy. It was just *waiting*. [[sn: This is a common class of bug: a safety mechanism firing on a false alarm. The watchdog was not wrong to exist — it was tuned for a task whose steps finish in seconds, not one whose steps take a quarter of an hour.]]

The root cause was not a typo or a bad number. It was that **synchronous training is a bad fit for long, multi-step agent rollouts** (a *rollout* is one full attempt — the whole command-and-reply conversation from start to finish). When each rollout is a many-command conversation lasting fifteen minutes, a lockstep system always leaves half its machinery idle long enough to trip a crash-detector.

The fix was not to patch the timer — it was to change the *shape* of the system. The team switched to **prime-rl**, a trainer that is **asynchronous**: the inference engines never sit idle, because they always generate the next batch of attempts while the previous batch is being graded and learned from. No idle engine means no false crash. The problem vanished — not because the math changed, but because the plumbing finally matched the job.

[[note: production || This is what "real ML is messy" looks like up close. The grown-up systems that train large agents are almost all asynchronous for exactly this reason: agent tasks are long and ragged, and you cannot afford to have expensive GPUs standing around waiting in lockstep. Choosing the right training *system* for the *shape* of your task is as much of the work as choosing the algorithm.]]

## The results, framed honestly

Now the numbers — and this section matters, because it is easy to tell this story dishonestly, and we are not going to.

The research paper reports these pass@1 targets on TerminalBench-2.0, when ECHO's extra loss is added on top of plain GRPO: the **8B** model going from **2.70 → 5.17**, and the **14B** model from **5.17 → 10.79**. Read those as what they are: on 89 tasks, `5.17` is under five tasks solved and `10.79` is under ten. These are small absolute numbers.

[[fig: A hand-drawn before/after bar comparison titled "ECHO vs plain GRPO (pass@1 on 89 tasks)". Two labeled pairs. LEFT pair "8B model": a short red bar "GRPO = 2.70" and beside it a green bar about twice as tall "ECHO = 5.17". RIGHT pair "14B model": a red bar "GRPO = 5.17" and a green bar about twice as tall "ECHO = 10.79". A curved orange arrow spanning each pair labeled "roughly DOUBLED". Below, a black note "same everything — only the extra loss added (a clean A/B)". A red dashed caption at the bottom in black: "but these are only a few of 89 tasks — small absolute numbers". Dashed takeaway box: "trust the RELATIVE doubling, not the exact number". || The honest headline: in a controlled comparison, adding ECHO's extra loss roughly doubles GRPO's score at both sizes. The absolute numbers are small and depend on private training data; the *relative* effect is the real claim.]]

So what should you believe? The trustworthy claim is the **relative, controlled** one: in a head-to-head where *everything is identical* except the one extra loss term, ECHO roughly **doubles** GRPO's score at both model sizes. That is a clean A/B test, and it is the real result — the extra loss helps, measurably.

The **absolute** numbers (`5.17`, `10.79`) are a different matter. They depend on roughly **6,000 private training tasks** that were never released, which nobody outside the original team can reproduce. So an honest replication does not promise to hit `10.79` on the nose. It reports landing *near the ballpark, with a gap* — and says so plainly. [[sn: The training tasks a replication *can* use come from open corpora — for example a collection called endless-terminals with about **3,255** tasks — but not the same 6,000 the paper used. Different training data, different absolute scores. The A/B effect is what carries across.]] The lesson is the *idea* and the *controlled comparison*, not a leaderboard position.

Where does the replication stand today? Honestly: the ECHO idea is implemented, and the extra loss has been checked to actually fire as intended [[sn: This check matters more than it sounds. For the extra loss to do anything, the terminal's replies must sit *inside* the span of text the model is scored on predicting — a naive wiring where the replies land only in the *next* prompt makes ECHO silently do nothing. Always verify a new loss is genuinely active, not just present in the code.]] rather than silently doing nothing. The small smoke test passed. The full large-scale runs are the ongoing work. That is the true status — a promising idea, cleanly compared, not a finished result — and stating it that way is the point.

We have now seen all three projects: a bug-fixer on a laptop, a real-code trainer on cloud GPUs, and a terminal agent that grows a world model for free. In the next chapter we step back and ask what they add up to — and where teaching machines to code goes from here.
