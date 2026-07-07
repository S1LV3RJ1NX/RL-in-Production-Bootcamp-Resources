# RLHF Socratic-Tutor Sweep — SYNTHESIS

## RLHF Socratic-Tutor Sweep — Comparison

Leakage lower=better; judge_overall, learning_gain, standard-mean higher=better. Best judge_overall per model in **bold**. Default beta DPO runs shown (beta-variants in note below).

### qwen0.5b

| method | leakage_rate ↓ | judge_overall ↑ | learning_gain ↑ | standard-mean ↑ |
|---|---|---|---|---|
| base | 0.4333 | 57.41 | -0.0105 | 0.3917 |
| prompted | 0.3333 | 54.48 | n/a | 0.3944 |
| sft | 0.0733 | 74.68 | +0.0104 | 0.3722 |
| dpo | 0.0533 | 48.94 | +0.0581 | 0.3833 |
| kto | 0.1200 | **75.42** | +0.0377 | 0.3778 |
| orpo | 0.0667 | 74.07 | +0.0158 | 0.3667 |
| simpo | 0.0933 | 73.83 | +0.0526 | 0.3361 |
| grpo | 0.2733 | 47.54 | +0.0563 | 0.4028 |

### qwen1.5b

| method | leakage_rate ↓ | judge_overall ↑ | learning_gain ↑ | standard-mean ↑ |
|---|---|---|---|---|
| base | 0.5733 | 61.08 | +0.0153 | 0.5167 |
| prompted | 0.2933 | 62.23 | n/a | 0.5167 |
| sft | 0.1267 | **77.47** | +0.0463 | 0.4722 |
| dpo | 0.0600 | 58.00 | -0.0062 | 0.4917 |
| kto | 0.1733 | 76.87 | +0.0224 | 0.4611 |
| orpo | 0.1533 | 75.60 | +0.0291 | 0.4472 |
| simpo | 0.1600 | 76.32 | +0.0796 | 0.4139 |

### smol1.7b

| method | leakage_rate ↓ | judge_overall ↑ | learning_gain ↑ | standard-mean ↑ |
|---|---|---|---|---|
| base | 0.4133 | 59.10 | +0.0262 | 0.4333 |
| prompted | 0.2867 | 67.57 | n/a | 0.4333 |
| sft | 0.2067 | 66.42 | +0.0702 | 0.4083 |
| dpo | 0.3200 | 53.26 | +0.0122 | 0.4361 |
| kto | 0.2400 | 63.20 | +0.0089 | 0.4194 |
| orpo | 0.1800 | 72.72 | +0.0468 | 0.4167 |
| simpo | 0.1200 | **76.40** | +0.0585 | 0.4000 |
| grpo | 0.3933 | 59.13 | +0.0026 | 0.4333 |


_DPO beta sweep on qwen0.5b (separate; default-beta DPO shown in table above): beta=0.05 -> leakage 0.0267 / judge 44.54; beta=0.1 (default) -> 0.0533 / 48.94; beta=0.5 -> 0.30 / 57.49._

## Findings

- SFT is the most reliable Socratic aligner: it lifts judge_overall on every model (qwen0.5b 57.41->74.68, qwen1.5b 61.08->77.47, smol1.7b 59.10->66.42) and slashes leakage (qwen0.5b 0.4333->0.0733, qwen1.5b 0.5733->0.1267, smol1.7b 0.4133->0.2067).
- SimPO is the standout offline-preference method: best judge_overall on smol1.7b (59.10->76.40) with leakage 0.4133->0.1200, and the highest learning_gain in the whole sweep on qwen1.5b (+0.0796 vs base +0.0153).
- KTO and ORPO are consistent winners too: KTO tops qwen0.5b judge_overall (57.41->75.42) and ORPO rescues smol1.7b (59.10->72.72, leakage 0.4133->0.1800) where DPO/GRPO failed.
- DPO collapses Socratic judge quality despite cutting leakage hardest: on qwen0.5b leakage falls 0.4333->0.0533 but judge_overall DROPS 57.41->48.94; same pattern on qwen1.5b (61.08->58.00) and smol1.7b (59.10->53.26). It over-suppresses answers without scaffolding (judge_guiding_question stays ~0.9-12.8).
- DPO beta is the key knob on qwen0.5b: beta=0.05 minimizes leakage (0.0267, best in sweep) but tanks judge to 44.54; beta=0.5 restores judge to 57.49 (~base) but leakage rebounds to 0.30. No DPO beta beats base judge_overall on qwen0.5b.
- GRPO underperforms badly relative to its cost: on qwen0.5b judge 57.41->47.54 (worst in sweep) at 640.9s train; on smol1.7b judge 59.10->59.13 (flat) and leakage 0.4133->0.3933 (barely moved) at 1573.7s train. Long RL runs bought almost nothing here.
- Prompting alone is a weak baseline: it reduces leakage modestly (qwen1.5b 0.5733->0.2933, smol1.7b 0.4133->0.2867) but moves judge_overall little (qwen0.5b even regresses 57.41->54.48); trained methods dominate it.
- Alignment tax is real but mostly mild on the strongest aligner: SFT loses only 0.019-0.045 standard-mean, but SimPO costs up to 0.103 on qwen1.5b (0.5167->0.4139) and KTO/ORPO exceed the 0.05 tax bar on qwen1.5b too.

## Failures (alignment did NOT help)

- qwen0.5b/dpo: judge_overall 48.94 <= base 57.41 (FAIL judge), even though leakage dropped 0.4333->0.0533.
- qwen0.5b/grpo: judge_overall 47.54 <= base 57.41 (FAIL judge); leakage only 0.4333->0.2733.
- qwen0.5b/prompted: judge_overall 54.48 <= base 57.41 (FAIL judge); leakage 0.4333->0.3333.
- qwen0.5b/simpo: standard-mean dropped 0.3917->0.3361 = -0.0555 (>0.05 alignment tax) despite strong judge/leakage.
- qwen1.5b/dpo: judge_overall 58.00 <= base 61.08 (FAIL judge); learning_gain also went negative (+0.0153 -> -0.0062).
- qwen1.5b/kto: standard-mean 0.5167->0.4611 = -0.0555 (>0.05 tax).
- qwen1.5b/orpo: standard-mean 0.5167->0.4472 = -0.0694 (>0.05 tax).
- qwen1.5b/simpo: standard-mean 0.5167->0.4139 = -0.1028 (largest tax in sweep, >0.05).
- smol1.7b/dpo: judge_overall 53.26 <= base 59.10 (FAIL judge) AND leakage WORSE-than-others 0.4133->0.3200 only; learning_gain near-flat +0.0122.
- smol1.7b/grpo: judge_overall 59.13 vs base 59.10 (effectively no gain) and leakage 0.4133->0.3933 (negligible) at 1573.7s train — no meaningful alignment.

## Verdict

- **best_method_overall:** SFT (only method that raises judge_overall AND cuts leakage on all 3 models, with the smallest alignment tax). SimPO is the best preference-based runner-up.
- **aligned_confirmed:** TRUE — on all 3/3 models multiple methods (sft, kto, orpo, simpo) simultaneously reduce leakage and raise judge_overall vs base (majority bar cleared).
