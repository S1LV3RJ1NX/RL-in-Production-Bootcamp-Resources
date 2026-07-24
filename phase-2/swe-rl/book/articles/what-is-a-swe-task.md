This whole book has one goal: to teach a machine to do software engineering — and then to *show you*, three different ways, exactly how that works. But before we can teach a machine a job, you and I need to agree, in plain words, on what the job actually is. So let us start there, with no code experience assumed and nothing to look up. By the end of this short chapter you will be able to say precisely what a "software engineering task" is — and why it turns out to be a wonderful thing to teach a machine.

Here is the one-sentence version, which we will spend the rest of the chapter unpacking: **a software engineering task is a concrete change to some code, together with an automatic way to check whether the change worked.** Hold onto the second half of that sentence. It is the quiet hero of everything that follows.

## A real, tiny example

Imagine an online store. Somewhere inside its website is a small piece of code — a **function**, a named recipe the computer can run — whose job is to work out the price after a discount. A human wrote it, and they made a mistake.

[[fig: A hand-drawn "before" code card, pale-red fill, titled "the bug" in black. Inside, three handwritten monospace lines in purple: "def final_price(price, discount):", "    return price - discount", "# 20% off $100 -> returns $80? no: $80... wait". To the right, a red shopping receipt sketch showing "$100 item, 20% off" with a big red "CHARGED: $98" and a red circled cross. A red handwritten note with a dashed arrow pointing at the "- discount" line: "subtracts 20, not 20 PERCENT". Bottom dashed takeaway box in black: "the code runs fine — it is just quietly WRONG". || The bug does not crash. It runs happily and charges the wrong price — the most common kind of real-world bug.]]

The intent was "take twenty percent off." But the code says `return price - discount`, which simply subtracts the number 20 from the price. Ask it for the price of a \$100 item at 20% off and it returns \$98 — because 100 − 20 would be \$80, but the store passes the discount in as a percentage, and the two got tangled. [[sn: This is worth noticing: the program does not crash, throw an error, or turn red. It runs perfectly and produces a *wrong answer*. A huge share of real bugs are exactly this quiet — which is why we need an automatic way to catch them.]] Whatever the exact numbers, the point is the same: the code runs, but it is wrong. Customers get charged the wrong amount. That is a bug.

## The definition of done: a test

Now, how would anyone *know* the code is wrong — reliably, automatically, without a human squinting at every case? With a **test**: a second, tiny piece of code whose only job is to check the first one.

A test states an expectation and demands that it hold. For our function it might read:

```python
assert final_price(100, 20) == 80   # $100 at 20% off should be $80
```

The word `assert` means "insist that this is true." If `final_price(100, 20)` really does return `80`, the test **passes** silently — all is well. If it returns anything else, the test **fails**: the computer stops and complains loudly that reality did not match the expectation. Right now, with our buggy code, this test fails.

[[note: metaphor || A test is a smoke detector for code. It sits there doing nothing while everything is fine, then screams the instant something is wrong. You do not read every line of a program to find bugs, any more than you sniff every corner of your house for smoke — you install detectors and trust them to go off.]]

So a software engineering task, made fully concrete, has three parts: some **code**, a **test** that is currently failing, and a definition of *done* that needs no human judgment at all — **make the failing test pass, without breaking the tests that already passed.** For our example, "done" means changing that one line to `return price * (1 - discount / 100)`, after which the test passes and the customer is charged \$80.

[[fig: Two hand-drawn code cards side by side. LEFT card, pale-red fill, titled "before" in black, with handwritten purple monospace "def final_price(price, discount):" then "    return price - discount" and a small red cross beside it. RIGHT card, pale-green fill, titled "the fix" in black, with handwritten purple monospace "def final_price(price, discount):" then "    return price * (1 - discount/100)". A bold blue arrow labeled "the change" points from the left card to the right card. Below the right card, a small green check and the handwritten line "assert final_price(100,20) == 80  PASSES" in green. Orange emphasis note underneath the pair: "done = the test goes from red to green". || A software task, drawn out: change the code so a failing test turns into a passing one. That green check is the entire definition of success.]]

## Why this shape is a gift

Everything about that little task is ordinary to a programmer. What is *not* ordinary — what this whole book hinges on — is how perfectly it suits a machine that learns by trial and error.

Think about what we have. We have a goal (`the test passes`) that a computer can check by itself, in a fraction of a second, with a completely unambiguous answer: pass or fail, yes or no, 1 or 0. No committee, no opinion, no "well, it depends." [[sn: Contrast this with tasks like "write a beautiful poem" or "design a nice logo," where success is a matter of taste and there is no button that returns *correct* or *incorrect*. Code is unusual: correctness is often mechanically checkable. That single property is why reinforcement learning works so well here.]] The test is a built-in, automatic teacher that will grade any attempt we throw at it, for free, forever.

[[note: aha || Keep this sentence in your head for the rest of the book: **the tests are the teacher.** We will never sit and hand-label the "right" code for a machine. We will hand the machine a problem, let it try, and let the tests say pass or fail. That grade is the only teaching signal it ever needs.]]

## What real software engineering adds on top

Our example is a single function and a single test. Real software engineering is the same shape, scaled up and made messier. A real project is not one function but thousands of files. A bug is usually reported as an **issue** — a written complaint, like *"the checkout charges the wrong amount when a discount is applied"* — filed on a site like GitHub, where much of the world's open-source code lives. A human (or, increasingly, an AI) reads the issue, hunts through the code to find the culprit, makes a change, and proposes it as a **pull request**: a bundle of edits, offered up for review, that says "here is my fix." The project's existing tests — often thousands of them — run automatically against the proposed change. If they all pass, the fix can be accepted.

[[fig: A hand-drawn horizontal flow of five rounded boxes connected by blue arrows, titled "the shape of real SWE work". Box 1 (black): "ISSUE — 'checkout charges wrong amount'" drawn as a little report card. Box 2 (black): "FIND IT — search thousands of files" drawn as a magnifying glass over stacked file icons. Box 3 (purple): "EDIT — change the code" drawn as a small code card. Box 4 (black): "PULL REQUEST — propose the change". Box 5: a dark terminal window running "$ run all tests" with a green "1000 passed" below it and a green check. A dashed takeaway box on the right in black: "same three parts — code, tests, 'done' = tests pass — just BIGGER". Orange note under box 2: "this searching + editing is where an agent needs a terminal". || Real-world software engineering is our tiny example scaled up: an issue, a hunt through a large codebase, an edit, and a wall of tests that decide whether it is done.]]

Notice that nothing essential changed. There is still code, there are still tests, and *done* still means the tests pass. What grew is the size of the haystack and the number of steps: you must *find* the right place among thousands of files, often by poking around — running commands, reading files, running the tests to see what breaks. That poking-around happens in a **terminal**, which is the subject of the very next chapter, and it is the difference between a machine that can only *talk* about a fix and one that can actually *make* it.

[[note: production || The grown-up version of exactly this task is a benchmark called SWE-bench: real bugs pulled from real GitHub projects, each with the real tests that must pass. The best open coding agents — systems like DeepSWE — are measured by how many of these genuine, messy tasks they can resolve on their own. Our three projects are miniatures of that same idea, built so you can see every gear turn.]]

## Where this leaves us

We now have the one definition the whole book depends on. A software engineering task is a change to code with an automatic, unambiguous test of success. Because that success can be checked by a machine — pass or fail, no human in the loop — we can let a model *try*, grade it instantly, and reward what worked. That is the exact recipe for reinforcement learning, and it is why teaching machines to code has suddenly become so powerful.

Before we can teach the trying, though, we need to give our machine a place to work — a way to run commands, read files, and run those all-important tests. That place is the terminal, and it is where we go next.
