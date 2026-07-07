export const meta = {
  name: 'socratic-rlhf-sweep',
  description: 'Train + eval many RLHF configs for Socratic SLM alignment on Modal GPUs',
  phases: [
    { title: 'Baselines', detail: 'eval untuned base + prompting-only ablation per model' },
    { title: 'TrainEval', detail: 'one agent per (method x model x seed): train LoRA on Modal then full eval' },
    { title: 'Synthesize', detail: 'aggregate before->after table, alignment-tax, flag failures' },
  ],
}

const PROJ = '/Users/raj/Claude_Brain/RL-Production-Workshop/rlhf-lab'
const MODAL = `${PROJ}/.venv/bin/modal`

const A = args || {}
const MODELS = A.models || ['qwen0.5b', 'qwen1.5b', 'smol1.7b']
const METHODS = A.methods || ['sft', 'dpo', 'kto', 'orpo', 'simpo', 'grpo']
const SEEDS = A.seeds || [0]
const MAX_STEPS = A.max_steps || 120
const GRPO_STEPS = A.grpo_steps || 60

// ---- build the config matrix ----------------------------------------------
const baselines = []
for (const m of MODELS) {
  baselines.push({ run_id: `base_${m}`, model: m, method: 'base', train: false,
                   system_variant: 'neutral', do_lg: true })
  baselines.push({ run_id: `prompted_${m}`, model: m, method: 'prompted', train: false,
                   system_variant: 'socratic', do_lg: false })
}
const trained = []
for (const m of MODELS) for (const meth of METHODS) for (const s of SEEDS) {
  trained.push({
    run_id: `${meth}_${m}_s${s}`, model: m, method: meth, seed: s, train: true,
    beta: meth === 'simpo' ? 2.0 : 0.1,
    lr: meth === 'sft' ? 0.0002 : 0.00005,
    max_steps: meth === 'grpo' ? GRPO_STEPS : MAX_STEPS,
    system_variant: 'neutral', do_lg: true,
  })
}

// extra DPO beta ablation on the smallest model (cheap, strengthens the paper)
if (A.beta_ablation !== false) {
  for (const b of [0.05, 0.5]) {
    trained.push({ run_id: `dpo_qwen0.5b_b${b}_s0`, model: 'qwen0.5b', method: 'dpo',
                   seed: 0, train: true, beta: b, lr: 0.00005, max_steps: MAX_STEPS,
                   system_variant: 'neutral', do_lg: true })
  }
}

// ---- the per-config agent --------------------------------------------------
const EVAL_SCHEMA = {
  type: 'object',
  required: ['run_id', 'model', 'method', 'ok'],
  properties: {
    run_id: { type: 'string' }, model: { type: 'string' }, method: { type: 'string' },
    ok: { type: 'boolean' },
    leakage_rate: { type: ['number', 'null'] },
    asks_question_rate: { type: ['number', 'null'] },
    mean_vreward: { type: ['number', 'null'] },
    judge_overall: { type: ['number', 'null'] },
    judge_withholds_answer: { type: ['number', 'null'] },
    judge_guiding_question: { type: ['number', 'null'] },
    judge_scaffolding: { type: ['number', 'null'] },
    learning_gain: { type: ['number', 'null'] },
    cold_score: { type: ['number', 'null'] },
    post_tutoring_score: { type: ['number', 'null'] },
    standard: { type: 'object' },
    train_runtime_s: { type: ['number', 'null'] },
    error: { type: ['string', 'null'] },
  },
}

function runConfig(cfg, phase) {
  const adapterDir = cfg.train ? `/vol/runs/${cfg.run_id}/adapter` : ''
  const trainCmd = cfg.train
    ? `${MODAL} run modal_apps/train.py --method ${cfg.method} --model ${cfg.model} ` +
      `--run-id ${cfg.run_id} --seed ${cfg.seed} --beta ${cfg.beta} --lr ${cfg.lr} ` +
      `--max-steps ${cfg.max_steps}`
    : '(no training for this baseline)'
  const evalCmd =
    `${MODAL} run modal_apps/evals.py::run_eval --run-id ${cfg.run_id} ` +
    `--model ${cfg.model} --method ${cfg.method} --seed ${cfg.seed || 0} ` +
    `--adapter-dir "${adapterDir}" --system-variant ${cfg.system_variant} ` +
    (cfg.do_lg ? '' : '--no-do-lg ')
  const prompt =
`Run ONE experiment config for a Socratic-tutor RLHF sweep. All compute is on Modal GPUs; you just launch commands and parse JSON.

Working directory (cd here first): ${PROJ}

${cfg.train ? `STEP 1 - TRAIN (LoRA on Modal). Run with Bash timeout 600000 ms:
  cd ${PROJ} && ${trainCmd}
It prints a "TRAIN RESULT" JSON and writes the adapter to ${adapterDir}. Note train_runtime_s from it.
` : `This is a baseline config (${cfg.method}); no training.
`}
STEP ${cfg.train ? 2 : 1} - EVAL (Modal). Run with Bash timeout 600000 ms:
  cd ${PROJ} && ${evalCmd}
It prints an "EVAL SUMMARY" JSON block and writes results/eval_${cfg.run_id}.json.

Then return the EVAL SUMMARY fields as your structured output: run_id="${cfg.run_id}", model="${cfg.model}", method="${cfg.method}", ok=true, and the metric fields (leakage_rate, asks_question_rate, mean_vreward, judge_overall, judge_withholds_answer, judge_guiding_question, judge_scaffolding, learning_gain, cold_score, post_tutoring_score, standard object${cfg.train ? ', train_runtime_s' : ''}).
If any command fails after a genuine retry, return ok=false with the error text in "error". Do NOT fabricate numbers - only report what the commands printed.`
  return agent(prompt, { label: cfg.run_id, phase, schema: EVAL_SCHEMA })
}

// ---- run ------------------------------------------------------------------
phase('Baselines')
const baseRes = await parallel(baselines.map(c => () => runConfig(c, 'Baselines')))

phase('TrainEval')
const trainRes = await parallel(trained.map(c => () => runConfig(c, 'TrainEval')))

const all = [...baseRes, ...trainRes].filter(Boolean)
log(`collected ${all.length}/${baselines.length + trained.length} configs`)

// ---- synthesis ------------------------------------------------------------
phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  required: ['markdown_table', 'findings', 'failures'],
  properties: {
    markdown_table: { type: 'string' },
    findings: { type: 'array', items: { type: 'string' } },
    failures: { type: 'array', items: { type: 'string' } },
    best_method_overall: { type: 'string' },
    aligned_confirmed: { type: 'boolean' },
  },
}
const synth = await agent(
`You are synthesizing results of a Socratic-tutor RLHF sweep. The raw per-config metrics (JSON) are below.
For EACH model, the 'base' row is the BEFORE (untuned) and each method row is AFTER alignment.

Produce:
1) markdown_table: a clean comparison table grouped by model, columns = method, leakage_rate (lower=better), judge_overall (higher), learning_gain (higher), and the mean of the 'standard' benchmark scores (the alignment-tax proxy; higher=better-retained). Bold the best method per model on judge_overall.
2) findings: 4-8 crisp bullet strings on before->after (e.g. "DPO cut leakage on qwen0.5b from X to Y while judge_overall rose A->B").
3) failures: any config where alignment did NOT help (leakage not reduced vs base, or judge_overall not above base, or standard dropped >0.05 vs base = large alignment tax). Be skeptical and exact; cite numbers.
4) best_method_overall and aligned_confirmed (true only if at least one method clearly reduces leakage AND raises judge_overall vs base on a majority of models).

Also write your markdown_table + findings to ${PROJ}/results/SYNTHESIS.md via Bash.

RAW METRICS:
${JSON.stringify(all, null, 1)}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA })

return { n_configs: all.length, synth }
