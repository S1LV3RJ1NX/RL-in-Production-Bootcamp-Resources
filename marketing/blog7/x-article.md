# X Article (Long-Form) — Blog 7: RLHF

## Schedule

- **Date:** Tuesday, August 4, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**RLHF: Teaching a Language Model What "Good" Means**

---

## Article Body (~1,200 words)

A language model only ever learns one thing: predict the next token on a sea of internet text. That makes it fluent. It does not make it helpful. Ask a raw pretrained model a question and it might reply with more questions, because that is what the text it imitated tended to do.

So how do you get from fluent autocomplete to an assistant you would actually use? You cannot just write down the answer. In a game the score is given: chess has checkmate, Breakout has a counter, the pendulum from the last post had a measurable angle. For "explain gravity to a child" there is no number that says how good the answer was.

And every simple proxy you invent gets gamed. Reward length and you get padding. Reward keywords and you get keyword stuffing. Reward a confident tone and you get confidently wrong. Quality is a holistic human judgment with no closed form.

The whole idea of RLHF is one sentence: if you cannot write the reward, learn it from people.

---

### Comparisons, not scores

People are noisy when they score an essay 0 to 100 and stable when they say which of two essays is better. So the raw material of RLHF is not a table of scores. It is a pile of comparisons: a prompt, a winner the human preferred, and a loser.

To turn "winner should outscore loser" into a trainable loss, pass the score gap through a sigmoid and read it as the probability the winner wins. That is the Bradley-Terry model, the same math behind chess Elo. The loss is the negative log-likelihood of the human's actual choices:

$$\mathcal{L}(\varphi) = -\,\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\varphi(x, y_w) - r_\varphi(x, y_l)\big)\Big]$$

In code it is a single line, exactly binary cross-entropy with the label fixed to "the winner won":

```python
import torch
import torch.nn.functional as F

def bradley_terry_loss(r_chosen, r_rejected):
    # -log sigma(r_w - r_l): the Bradley-Terry negative log-likelihood
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

[EMBED IMAGE HERE: fig-bradley-terry.png — the sigmoid maps a score gap to a win probability; the gradient flattens once a pair is clearly correct]

The sigmoid buys one crucial behavior. Its gradient is 1 minus sigma, so a pair the model already gets right pulls almost nothing, and a pair it gets wrong pulls hard. The loss quietly retires the settled comparisons and spends its effort on the ones still in doubt. Trained one pass over Anthropic's HH-RLHF on top of GPT-2, preference accuracy climbs from 0.51 to about 0.59. Small model, noisy human labels, but well above the coin-flip line.

---

### The reward model is just a head on a transformer

What network produces that score? Reading a prompt and answer and judging it is a language task, so the backbone is another transformer. You take a pretrained model and change its output. A transformer emits a hidden state at every position, a rich vector summarizing everything up to there. The usual language head maps that to one logit per vocabulary word. A reward model swaps in a scalar head, a single vector that maps the last token's hidden state to one number.

Why the last token? Decoder attention is causal, so only the final position has read the entire prompt and answer. It is the one place to read a verdict on the whole response. Almost all of the reward model is the pretrained trunk. Only the tiny head, plus a little fine-tuning, learns human taste.

---

### One score at the end, a signal at every token

Here is the awkward gap. The reward model hands back one number for a whole answer. PPO updates one token at a time and needs a per-token signal. RLHF bridges it in three moves.

First, a per-token reward: a tiny KL toll on every token for drifting from the frozen reference, plus the reward-model verdict dropped on the final token only. Second, roll those rewards backward into a return, the reward-to-go from each step onward, so the terminal verdict flows back onto every earlier token. Third, subtract the critic to get the advantage:

$$A_t = G_t - V(s_t)$$

The critic is a value head, a second scalar head on the same trunk, trained during RL to predict the return from a half-written answer. On real GPT-2 rollouts its error drops tenfold and its predictions correlate 0.94 with realized returns. That is what turns one score into a signed nudge per token: positive advantage pushes a token up, negative pushes it down.

[EMBED IMAGE HERE: fig-per-token-credit.png — one reward at the last token becomes a spread of signed per-token advantages]

---

### RLHF says "KL" twice

This trips up almost everyone. There are two KL terms doing two different jobs.

The clip from PPO is a KL on step size. It keeps each update inside a trust region around the policy that collected this batch, refreshed every batch. The leash is a KL on distance from home. It sits in the reward and pulls the policy back toward the frozen reference over the whole run. The headline objective is the reward minus that leash:

$$\max_\theta \;\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta}\big[r_\varphi(x, y)\big] \;-\; \beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

The clip is a speed limit on each step. The leash is a fence around the neighborhood. A speed limit does not keep you in town, and a fence does not stop you crashing, so you need both.

---

### Why the leash is not optional

The reward model is a proxy trained on finite data, and PPO is very good at exploiting proxies. Turn the leash off (beta = 0) and the reward-model score climbs beautifully, from about +0.07 to +0.38. The catch: KL from the reference runs past 40, bigram diversity collapses to 0.31, and the answers degenerate into repeated tokens the reward model happens to love.

[EMBED IMAGE HERE: fig-ppo-ablation.png — beta=0.2 keeps KL in single digits; beta=0 climbs higher on reward while KL runs off past 40]

You can measure the exploitability directly. Craft degenerate variants of real answers and count how often each beats the genuine one under the frozen reward model: a repeated clause wins 46.6% of the time, stacked flattery 57.5%, pure padding 63.0%, and reward-model score correlates 0.25 with raw length. This is Goodhart's law: when a measure becomes a target, it stops being a good measure. The defenses are the KL leash, early stopping at the true-quality peak, reward normalization, and retraining the reward model on the policy's own outputs.

---

### Who this is for

If you have read "PPO aligned ChatGPT" and wanted every symbol earned rather than asserted, this is that walk. It maps text generation onto an MDP, learns a reward from comparisons, builds the value head and per-token advantages by hand on a worked example, and then runs the whole loop on GPT-2 with the ablation that proves the leash matters.

---

### What's next

Blog 8 is GRPO. RLHF holds four big models at once, and the most expensive and unstable is the value-head critic. The next post drops it: keep PPO's clip, but replace the learned critic with a group baseline, sample several answers per prompt and use their mean reward as the baseline. Same gradient, cheaper advantage, and the method behind today's reasoning models like DeepSeek-R1.

Full post with typed Python, the Bradley-Terry build, the per-token credit walk-through, and a runnable RLHF loop on GPT-2: https://prathameshsaraf.com/blogs/07-rlhf/

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the rest of the series.

---

## Header Image

- Use **`blog7-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, three neon nodes (REWARD MODEL, VALUE HEAD, KL LEASH) converging into a glowing "ALIGNED LLM" node, with the title and subtitle set on the right.
- Embed `fig-bradley-terry.png` in the "comparisons, not scores" section. X does not support SVG, so export the PNG first: `rsvg-convert -z 2 blogs/07-rlhf/images/fig-bradley-terry.svg -o marketing/blog7/fig-bradley-terry.png`.
- Embed `fig-per-token-credit.png` in the "one score at the end" section. Export with: `rsvg-convert -z 2 blogs/07-rlhf/images/fig-per-token-credit.svg -o marketing/blog7/fig-per-token-credit.png`.
- Embed `fig-ppo-ablation.png` in the "why the leash is not optional" section. Export with: `rsvg-convert -z 2 blogs/07-rlhf/images/fig-ppo-ablation.svg -o marketing/blog7/fig-ppo-ablation.png`.
- Optional inline image: `fig-goodhart.png` near the Goodhart paragraph.
- Fallback header: `ai-hero.png` from `blogs/07-rlhf/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. You cannot write reward(text) for 'be helpful,' so you learn it: humans compare answers, a reward model fits those comparisons, and PPO climbs it on a KL leash. Drop the leash and it games the reward instead of improving."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "You cannot write a Python function that scores 'be helpful and honest.' RLHF is the workaround the whole field standardized on. Here is every symbol of it." (recommended: pattern interrupt plus open loop)
2. "RLHF says 'KL' twice and they are not the same thing. One caps each step, one caps how far you drift from home. Confuse them and you cannot explain reward hacking."
3. "Turn off RLHF's KL leash and the reward score goes up while the answers turn to repeated-token mush. Higher score, worse text. Goodhart's law, measured."
4. "One reward at the last token becomes a signed nudge for all 200. That trick, value head plus reward-to-go, is the quiet heart of RLHF."

Then reply to anyone who engages, same as the first hour of the original.
