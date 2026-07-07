export const meta = {
  name: 'socratic-rlhf-sweep2',
  description: 'Pass 2: seed replication (error bars) + 4th model (SmolLM2-360M)',
  phases: [
    { title: 'Seeds', detail: 'seeds 1,2 for the 3 main models -> error bars' },
    { title: 'Model4', detail: 'SmolLM2-360M: base + prompted + 6 recipes' },
  ],
}

const PROJ = '/Users/raj/Claude_Brain/RL-Production-Workshop/rlhf-lab'
const MODAL = `${PROJ}/.venv/bin/modal`

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
    learning_gain: { type: ['number', 'null'] },
    standard: { type: 'object' },
    error: { type: ['string', 'null'] },
  },
}

function runConfig(cfg, phase) {
  const adapterDir = cfg.train ? `/vol/runs/${cfg.run_id}/adapter` : ''
  const trainCmd = cfg.train
    ? `${MODAL} run modal_apps/train.py::main --method ${cfg.method} --model ${cfg.model} ` +
      `--run-id ${cfg.run_id} --seed ${cfg.seed} --beta ${cfg.beta} --lr ${cfg.lr} ` +
      `--max-steps ${cfg.max_steps}`
    : '(baseline, no training)'
  const evalCmd =
    `${MODAL} run modal_apps/evals.py::run_eval --run-id ${cfg.run_id} ` +
    `--model ${cfg.model} --method ${cfg.method} --seed ${cfg.seed || 0} ` +
    `--adapter-dir "${adapterDir}" --system-variant ${cfg.system_variant} ` +
    (cfg.do_lg ? '' : '--no-do-lg ')
  const prompt =
`Run ONE experiment config for the Socratic-tutor RLHF sweep (pass 2). Compute is on Modal GPUs; you launch commands and parse JSON. cd ${PROJ} first.

${cfg.train ? `STEP 1 - TRAIN (Bash timeout 600000 ms):
  cd ${PROJ} && ${trainCmd}
Writes a LoRA adapter to ${adapterDir} and prints TRAIN RESULT JSON.
` : `Baseline config (${cfg.method}); no training.
`}
STEP ${cfg.train ? 2 : 1} - EVAL (Bash timeout 600000 ms):
  cd ${PROJ} && ${evalCmd}
Prints an "EVAL SUMMARY" JSON block and writes results/eval_${cfg.run_id}.json.

Return the EVAL SUMMARY fields as structured output (run_id="${cfg.run_id}", model="${cfg.model}", method="${cfg.method}", ok=true, plus the metric fields). On genuine failure return ok=false with the error text. Never fabricate numbers.`
  return agent(prompt, { label: cfg.run_id, phase, schema: EVAL_SCHEMA })
}

// ---- pass-2 matrix: ONLY new configs (seed 0 of the 3 main models already done)
const MAIN = ['qwen0.5b', 'qwen1.5b', 'smol1.7b']
const FAST = ['sft', 'dpo', 'kto', 'orpo', 'simpo']

const seedCfgs = []
for (const m of MAIN) for (const meth of FAST) for (const s of [1, 2]) {
  seedCfgs.push({ run_id: `${meth}_${m}_s${s}`, model: m, method: meth, seed: s, train: true,
    beta: meth === 'simpo' ? 2.0 : 0.1, lr: meth === 'sft' ? 0.0002 : 0.00005,
    max_steps: 120, system_variant: 'neutral', do_lg: true })
}
for (const m of MAIN) {  // one extra GRPO seed (reduced steps, H100)
  seedCfgs.push({ run_id: `grpo_${m}_s1`, model: m, method: 'grpo', seed: 1, train: true,
    beta: 0.1, lr: 0.00005, max_steps: 40, system_variant: 'neutral', do_lg: true })
}

const m4 = [
  { run_id: 'base_smol360m', model: 'smol360m', method: 'base', train: false, system_variant: 'neutral', do_lg: true },
  { run_id: 'prompted_smol360m', model: 'smol360m', method: 'prompted', train: false, system_variant: 'socratic', do_lg: false },
]
for (const meth of ['sft', 'dpo', 'kto', 'orpo', 'simpo', 'grpo']) {
  m4.push({ run_id: `${meth}_smol360m_s0`, model: 'smol360m', method: meth, seed: 0, train: true,
    beta: meth === 'simpo' ? 2.0 : 0.1, lr: meth === 'sft' ? 0.0002 : 0.00005,
    max_steps: meth === 'grpo' ? 40 : 120, system_variant: 'neutral', do_lg: true })
}

phase('Seeds')
const r1 = await parallel(seedCfgs.map(c => () => runConfig(c, 'Seeds')))
phase('Model4')
const r2 = await parallel(m4.map(c => () => runConfig(c, 'Model4')))

const ok = [...r1, ...r2].filter(Boolean).filter(r => r.ok)
log(`pass-2: ${ok.length}/${seedCfgs.length + m4.length} configs ok`)
return { ok: ok.length, total: seedCfgs.length + m4.length }
