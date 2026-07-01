# Daily Follow-ups — Blog 5: Policy Gradients

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Jul 22, then one per day through Follow-up 6 = Mon Jul 27. On Tuesday Jul 28 the series moves to blog 6.

Blog link: https://prathameshsaraf.com/blogs/05-policy-gradients/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #RLHF #LLMs #LearningInPublic

---

## Follow-up 1 (Wed Jul 22) — delete the middleman

Value methods learn a number for every action, then pick moves with an argmax. There is a quieter cost to that: to act, you need either the argmax or the environment model.

Policy gradients drop both. You parameterize the policy and nudge its parameters to make good actions more likely. No argmax, no model, just gradient ascent on expected reward.

That pivot is the branch of RL that leads to PPO and RLHF. What made you switch from value methods to policy methods?

Link in comments.

---

## Follow-up 2 (Thu Jul 23) — the score-function trick

Policy gradients hang on one identity from calculus.

We want to differentiate an average over actions, but the actions are drawn from the very distribution we are changing, so we cannot move the gradient inside. The trick: the gradient of a probability equals the probability times the gradient of its log. That turns the gradient back into an average we can sample.

grad J = E[ R(a) * grad log pi(a) ]

Sample an action, see its reward, push that action up in proportion to how good it was. That one line is REINFORCE. Where did the log first confuse you?

Link in comments.

---

## Follow-up 3 (Fri Jul 24) — one reward, every logit moves

A softmax policy over nine aiming angles. You sample one angle, see one reward, and update once.

What happens to the other eight angles you did not try?

They all move. The angle you tried gets pushed up, the other eight get pushed down, and the pushes sum to exactly zero. Softmax probabilities add to one, so raising one must lower the rest. The whole REINFORCE update fits in that one sentence.

Link in comments.

---

## Follow-up 4 (Sat Jul 25) — why subtract a baseline

Plain REINFORCE works, but it is loud. A single sampled trajectory gives a wildly swinging gradient, so learning crawls.

The fix is to scale by how much better the reward was than expected, not by the raw reward. Subtract the value of the state and you get the advantage:

A = G - V(s)

When rewards are mostly positive, the raw version pushes every action up and the policy drifts without separating good from bad. The baseline centers it: better-than-average goes up, worse-than-average goes down. Same expectation, far less noise.

Link in comments.

---

## Follow-up 5 (Sun Jul 26) — credit assignment

In a bandit there is no future, so the weight on an action is just its reward. Add states and that breaks.

A quiet "step closer" that costs you a little now can set up a big reward later. So the weight becomes the reward-to-go, the discounted sum of everything that follows the step. A move that looked like a small loss gets credited for the win it enabled.

That is credit assignment, and it is why an end-of-episode reward can still teach the very first move. How do you think about credit in your own training runs?

Link in comments.

---

## Follow-up 6 (Mon Jul 27) — the variance ladder, and the week in recap (attach fig-variance-ladder)

Every trick in policy gradients aims at one enemy: the variance of the gradient estimate. Here is the payoff in numbers, from the Archer MDP.

- REINFORCE: gradient variance 0.0437, greedy return 4.66
- Plus a baseline: variance 0.0006, return 9.60
- Actor-Critic: variance 0.0004, return 9.62

The baseline cuts variance by orders of magnitude, and only the low-variance methods actually solve the task. Lower variance is not cosmetic, it is the line between learning and not.

So here is blog 5 in five lines:

- Skip values, optimize the policy directly by gradient ascent.
- The score-function trick turns a gradient you cannot sample into an average you can.
- One reward moves every logit, and the pushes sum to zero.
- A baseline centers the signal and kills the variance.
- Actor-Critic learns the baseline as a critic, the template every modern method reuses.

Tomorrow the series moves to blog 6: TRPO and PPO, where we stop each update from taking too big a step and collapsing the policy. That is the last piece before the RLHF stack. Have you hit unstable policy updates in your own runs?

(Attach: fig-variance-ladder.png)

Link in comments.

---

## Notes

- Vary the opening line on reuse; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep equations readable (for example "grad J = E[ R * grad log pi(a) ]" and "A = G - V(s)").
- If a policy-gradient, PPO, or RLHF paper trends this week, quote-post with "the policy-gradient foundation behind this" plus your link.
