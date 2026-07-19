# X Article (Long-Form) — Blog 11: Dreaming to Dodge (World Models)

## Schedule

- **Date:** Tuesday, September 1, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**Dreaming to Dodge: Training a Doom Agent Entirely Inside Its Own Dream**

---

## Article Body (~1,300 words)

I trained a Doom agent that never played the real game. It practiced dodging fireballs inside a neural network's dream, then walked into the actual game and matched a hand-coded oracle that reads the real screen.

The dream was the easy part. The whole project turned on three failures the dream produced along the way: it erased the fireball, it refused to let anyone die, and the policy learned to cheat it.

This is the final post in the series: Ha and Schmidhuber's famous 2018 result (train a VizDoom dodger inside a learned dream) reproduced from scratch with the 2023 IRIS recipe, on rented cloud GPUs, with a paper and a playable neural simulator at the end.

---

### A language model for pixels

A world model is a network that learned to simulate an environment. Give it the current observation and an action and it predicts the next observation, the reward, and whether the episode ends. Once it dreams plausibly, you train the policy inside the dream. Real experience gets spent once, on the model; after that, practice steps are generated, parallel, and safe.

IRIS makes this concrete with a move borrowed from NLP. A VQ-VAE tokenizer turns each 64x64 frame into 64 discrete tokens from a 512-word codebook. A GPT-style Transformer then predicts the game one token at a time, exactly like a language model predicts the next word. The context is an interleaved ribbon of frame tokens and actions:

$$16 \times (64 + 1) = 1040$$

Sixteen timesteps, 64 frame tokens plus 1 action each. That ribbon is the model's entire visible universe, about a second and a half of Doom.

The task is take_cover: monsters lob fireballs, you can only strafe left or right, +1 per step you survive. Return equals survival time. Death lands on about 1% of steps. Hold those two facts; they each break a component.

---

### Failure one: the vanishing fireball

Train the tokenizer with a plain uniform pixel loss and measure how much fireball brightness survives reconstruction: 0.53. Half the threat is gone, because the fireball is a few dozen pixels in a 4,096-pixel frame, and erasing it barely moves an average. A dream with pre-dimmed fireballs cannot teach dodging.

The obvious fix, weight bright pixels more, stalled at 0.63. The walls in take_cover are bright too, and they soaked up the extra votes. What actually distinguishes a fireball is not brightness. It is warmth: red-orange in a gray and brown room. Weighting pixels by how much red exceeds green and blue took recall to 0.95.

```python
# per-pixel weight: brightness + warmth (R above G,B => fireballs)
lum = x.amax(dim=1, keepdim=True)
warm = (x[:, :1] - 0.5 * (x[:, 1:2] + x[:, 2:3])).clamp(min=0.0)
w = 1.0 + self.cfg.fg_weight * lum + self.cfg.warm_weight * warm
err = (x_hat - x).abs().mean(dim=1, keepdim=True)
recon = (w * err).sum() / w.sum()
```

[EMBED IMAGE HERE: fig-fireball-recall.png — uniform 0.53, luminance 0.63, warmth 0.95]

The lesson generalizes past Doom: a uniform objective defines "important" as "occupies many pixels," and it will quietly discard the pedestrian at the edge of the frame, the tumor a few pixels wide, the one indicator light that turned red. Somebody has to tell the representation what to keep.

---

### Failure two: the model that refused to dream death

The trained world model produced gorgeous, controllable Doom. And its dreams never ended. Roll them a thousand steps and nobody dies.

Death is 1% of steps, an 89-to-1 class imbalance, so the done head discovered that predicting "alive" every time costs almost nothing under an unweighted loss. In a survival task that is fatal in the literal sense: return equals survival time, so termination is the entire training signal. A dream where nobody dies makes every policy look equally good.

The fix is a 55x class weight on the death class. Death recall went from 0 to 1.0. Same disease as the fireball, one level up the stack: a rare event that a uniform objective rounds away.

---

### Failure three: the policy cheated the dream

With fireballs in the tokens and death in the dream, I trained an LSTM actor-critic on imagined rollouts. Imagined return climbed beautifully. Real performance came back worse than random.

The agent had learned to hold one direction forever. The world model had a soft spot that scored wall-hugging as safe, and a gradient policy with 0.8M parameters found it and moved in. Inside the dream, the cheat out-survived the reactive oracle, 55 steps to 46. In reality it collapsed to 45 while the oracle did 98.3. If you know reward hacking from RLHF, this is the same event with the world model playing the gameable reward model.

[EMBED IMAGE HERE: fig-exploitation-gap.png — the dream ranks the cheat above the oracle; reality inverts it]

A diagnostic saved the project: roll the dream under canned policies and count survival. The oracle lasted 46 in-dream against about 30 for any fixed strafe. So the dream itself rewards dodging; the training signal existed. The problem was the policy's power to exploit, and how I selected what to deploy.

The recipe that closed it, straight from the 2018 playbook plus two modern twists:

- A controller too small to cheat: a 1,795-parameter MLP, evolved by CMA-ES on dream survival. Not enough capacity to memorize the model's soft spots, enough to express "move away from the threatening column."
- A feature that transfers: dream tokens and real tokens have different statistics, so the controller reads the decoded image instead, pooled to 54 numbers. Same view in both worlds.
- Selection the dream cannot inflate: run six independent CMA searches and pick the winner on held-out real episodes. An earlier controller scored 75 on its selection seeds and 50 on fresh ones. Selection optimism is real.

---

### The result

On held-out episodes, the dream-trained agent survives 96.6 steps: above random (67), above a model-free Double-DQN trained on the same 200,000 real frames (90), statistically matching the oracle (98.3). Its best episodes clear the original World Models "solved" bar of 188 steps. Zero real-environment gradients anywhere in training.

[EMBED IMAGE HERE: fig-survival-showdown.png — the survival showdown with the 188-step solved bar]

The DQN bar is the one to stare at. Same real-data budget, two ways to spend it: direct model-free learning got 90, building a dream and practicing inside it got 96.6.

And the claim is playable. At dreaming-to-dodge.vercel.app there is no game engine; the Transformer predicts every frame from your strafes, and a "watch the AI" mode lets the trained controller play inside its own dream.

---

### Who this is for

If you train policies on top of any learned model (a world model, a reward model, a simulator), this post is a field guide to the gap between "the model is accurate" and "the thing trained inside it works in reality." The three leaks (capacity, transfer, selection) each needed their own fix, and none of them showed up in the model's own metrics.

The sentence I kept coming back to: the world model is the easy part; getting a policy to learn a robust skill inside an imperfect model, without exploiting it, is the hard part.

Full post with the VQ-VAE from basics, the straight-through estimator, the codebook-collapse toy, the KV cache, and the Modal harness: https://prathameshsaraf.com/blogs/11-world-models/

Learning RL for LLMs through the @VizuraAI bootcamp. The full series, from the Bellman equation to this, is on the same site.

---

## Header Image

- Use **`blog11-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, neon nodes (FRAME, TOKENIZER, WORLD MODEL, CONTROLLER) looping into a glowing "DREAM" node, with the title and subtitle set on the right.
- Embed `fig-fireball-recall.png` in the vanishing-fireball section.
- Embed `fig-exploitation-gap.png` in the cheating-policy section.
- Embed `fig-survival-showdown.png` in the results section. This is the headline figure.
- Optional inline image: `fig-dream-rank.png` next to the "the dream rewards dodging" paragraph.
- Fallback header: `ai-hero.png` from `blogs/11-world-models/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. The agent saw its first real fireball at evaluation time and dodged it, because it had already dodged thousands of imagined ones. The hard part was stopping it from cheating the imagination."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "My policy found a strategy that out-survived the oracle inside the dream (55 vs 46 steps) and collapsed in the real game. Model-based RL's dirty secret is that the policy is an adversary of its own world model." (recommended: pattern interrupt plus open loop)
2. "An agent trained with zero real-environment gradients survived 96.6 steps in Doom. A DQN trained on the same 200,000 real frames managed 90."
3. "The tokenizer kept 53% of the fireball. Weighting bright pixels got 63%, because the walls are bright too. Weighting warm pixels got 95%. The loss decides what the dream contains."
4. "There is no game engine behind this page. A Transformer predicts every frame from your strafes, and the agent that plays it was trained entirely inside: dreaming-to-dodge.vercel.app"

Then reply to anyone who engages, same as the first hour of the original.
