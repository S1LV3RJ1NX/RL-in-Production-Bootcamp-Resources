# X Article (Long-Form) — Blog 11: World Models

## Schedule

- **Date:** Tuesday, September 1, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**World Models: Training an Agent Entirely Inside Its Own Dream**

---

## Article Body (~1,300 words)

I trained an RL agent that never played the real game. It practiced inside a neural network's dream of the game, then went out and caught the ball 500 times out of 500 in reality, matching a hand-coded oracle.

The first version of that agent played at random. The bug was four components upstream of the agent, in a loss function that had quietly deleted the ball from the dream. Finding it taught me more than the win did.

This is the final post in the series: IRIS, the 2023 "world models as language models" architecture, rebuilt from scratch in three small networks, run honestly, debugged in public, then scaled to VizDoom.

---

### A language model for pixels

A world model is a network that learned to simulate an environment. Give it the current observation and an action and it predicts the next observation, the reward, and whether the episode ends. Once it dreams plausibly, you can train a policy inside the dream. Real experience gets spent once, on the model. After that, the policy's millions of practice steps cost almost nothing and risk nothing.

IRIS makes this concrete with a move borrowed from NLP. A VQ-VAE tokenizer turns each 64x64 frame into 16 discrete tokens from a 256-word codebook. A GPT-style Transformer then predicts the environment one token at a time, exactly like a language model predicts the next word. An actor-critic watches decoded dream frames and picks actions.

The testbed is pixel Catch: a paddle, one falling ball, +1 for a catch, -1 for a miss. Small enough that every failure is visible. That mattered more than I expected.

---

### The world model was never the problem

The scary-sounding component, the Transformer that "learns physics," trained in 48 seconds to 98.2% next-token accuracy. Rolled forward from one real frame, its dream diverged from reality by a pixel MSE of about 0.02. Frame for frame, the dream and the game were nearly the same picture.

Why so easy? Next-token prediction over a 256-way discrete space, in a deterministic environment, with a correct label at every position, is the most favorable setup a Transformer can be handed. It is ordinary supervised learning. The policy, meanwhile, gets one sparse scalar at the end of an episode.

So when the fully assembled agent finished 500 evaluation episodes with a catch rate of 0.11, statistically identical to random, the world model looked innocent. It was, and it wasn't.

$$10 \times (16 + 1) = 170$$

That is the Transformer's whole universe: ten timesteps of 16 frame tokens plus 1 action token each. The model is 98% accurate at reproducing what the tokenizer kept. Nobody promised the tokenizer kept the ball.

---

### The vanishing ball

Render the tokenizer's reconstructions and there is the smoking gun: the paddle is redrawn perfectly, and the ball is gone. The dream was a game of Catch with nothing to catch. Random play is the correct answer to a game with no visible objective.

The obvious diagnosis is codebook collapse, and it was real: the tokenizer used 3 of its 256 codes, all three describing background. Dead codes get exactly zero gradient (they appear in the loss zero times), so collapse is a stable rich-get-richer equilibrium, and the textbook cures work. EMA codebook updates plus dead-code revival took usage from 3 codes to 254.

Then the controlled experiment ruined the story. With a fully healthy codebook, across 11 random seeds, the ball survived reconstruction in 2 of 11 runs. The collapsed codebook had managed 4 of 11. Fixing the codebook did not bring the ball back at all.

The real culprit was the reconstruction loss. The ball is about 0.5% of the pixels. Under a uniform per-pixel average, erasing it entirely barely moves the number; the reconstruction error stays excellent (0.0165) with the one task-critical object missing. Keeping the ball and dropping the ball cost nearly the same, so seed noise decided. A coin flip, which is exactly what 2 of 11 looks like.

The fix is one line. Weight each pixel by its brightness:

```python
# w_p = 1 + 25 * brightness: a ball pixel is worth 26 votes
w = 1.0 + self.cfg.fg_weight * x.amax(dim=1, keepdim=True)
err = (x_hat - x).abs().mean(dim=1, keepdim=True)
recon = (w * err).sum() / w.sum()
```

The ball's share of the objective goes from 0.5% to about 13%. Across the same 11 seeds: 11 of 11, recall 1.00, zero variance. When the variance vanishes, you fixed the cause, not a symptom.

[EMBED IMAGE HERE: fig-ball-recall.png — the dissociation: codebook fix 2/11, loss fix 11/11]

---

### One upstream fix, noise to oracle

With the ball back in the tokens, I changed nothing about the world model or the actor-critic and retrained. The same policy that flatlined for 500 updates crossed zero imagined return around update 180 and kept climbing.

Greedy evaluation on 500 fresh real episodes: catch rate 1.00, mean return +1.00, standard deviation 0.0. Identical to the hand-coded oracle that always moves toward the ball's column.

[EMBED IMAGE HERE: fig-catch-showdown.png — 0.11 in the random band, then 1.00 matching the oracle]

Only the first frame of any training rollout was real. The ball it learned to track was invented by the Transformer, the catches were declared by the model's own reward head, and the skill transferred to reality anyway.

---

### Scaling the dream to Doom

The same skeleton, pointed at VizDoom's take_cover (dodge fireballs, +1 per surviving step), is where the lessons compound. Nothing went smoothly, and each failure had the same shape as Catch's.

The tokenizer smoothed the fireball into the wall until a warmth term (upweight pixels where red beats green and blue) took fireball recall from 0.53 to 0.95. The done head never predicted death (about 1% of steps, an 89-to-1 imbalance) until a class weight took death recall from 0 to 1.0. Uniform objectives keep dropping the rare thing that matters.

And one failure Catch could not teach: the policy learned to exploit the world model. Degenerate strategies (hold one direction forever) out-survived the oracle inside the dream and collapsed in reality. Reward hacking, with the world model as the gameable reward. Longer horizons, more collect-train rounds, and held-out real-episode selection closed the gap.

Final numbers, on held-out episodes: the dream-trained agent survives 96.6 steps against 67 for random, 90 for a model-free DQN trained on 200,000 real frames, and 98.3 for a reactive oracle that sees the real game. Zero real-environment gradients.

[EMBED IMAGE HERE: fig-doom-survival.png — the survival showdown]

---

### Who this is for

If you train policies on top of any learned representation (a VQ tokenizer, an encoder, a reward model), this post is a field guide to the failure mode where every component reports healthy metrics and the system learns nothing. The debugging discipline transfers: walk upstream from the symptom, and distrust any diagnosis you haven't isolated with a one-variable experiment. Codebook collapse was real, curable, and not the cause.

The sentence I keep coming back to: a world model can only teach a policy what its tokenizer chooses to preserve. Somebody has to tell the representation what matters, because a uniform loss defines "important" as "numerous," and the ball never is.

Full post with the straight-through estimator, the EMA and revival code, lambda-returns worked by hand, the Modal harness, and the Doom capstone: https://prathameshsaraf.com/blogs/11-world-models/

Learning RL for LLMs through the @VizuraAI bootcamp. The full series, from the Bellman equation to this, is on the same site.

---

## Header Image

- Use **`blog11-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, neon nodes (TOKENIZER, WORLD MODEL, ACTOR-CRITIC) looping into a glowing "DREAM" node, with the title and subtitle set on the right.
- Embed `fig-ball-recall.png` in the vanishing-ball section (the dissociation result).
- Embed `fig-catch-showdown.png` in the noise-to-oracle section. This is the headline figure.
- Embed `fig-doom-survival.png` in the Doom section.
- Optional inline image: `fig-policy-curves.png` next to the "crossed zero around update 180" paragraph.
- Fallback header: `ai-hero.png` from `blogs/11-world-models/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. The world model was 98% accurate and the agent still learned nothing, because it was a faithful model of a world whose tokenizer had already deleted the ball. Fix the loss one line upstream and the same policy goes from random to matching an oracle."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "My world model hit 98.2% accuracy and the agent trained inside it played at random. Both facts were true at once, and the reason is the most useful bug I've ever chased." (recommended: pattern interrupt plus open loop)
2. "I fixed codebook collapse across 11 seeds and the ball came back in 2 of them. The fix everyone reaches for was real, curable, and not the cause."
3. "The ball is 0.5% of the pixels. A uniform reconstruction loss will delete it and still report an excellent error. One line reweights it to 13% of the vote and the agent goes from random to oracle."
4. "An agent trained with zero real-environment gradients survived 96.6 steps in Doom. A DQN trained on 200,000 real frames managed 90."

Then reply to anyone who engages, same as the first hour of the original.
