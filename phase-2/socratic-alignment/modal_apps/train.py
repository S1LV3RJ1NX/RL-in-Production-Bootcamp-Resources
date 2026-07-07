"""Train one Socratic-tutor config with TRL + LoRA on Modal.

Methods: sft | dpo | kto | orpo | simpo | grpo   (ppo lives in ppo.py)
Each trains the base instruct model independently (so every method is compared
against the same untuned base). Saves a LoRA adapter to /vol/runs/<run_id>/.

Run one:  modal run modal_apps/train.py --method dpo --model qwen0.5b --run-id smoke_dpo
"""
import json
import modal
from common import (
    APP_PREFIX, TRAIN_IMAGE, VOLUMES, VOL_PATH, BASE_MODELS, GPU_SMALL,
    TUTOR_SYSTEM, verifiable_reward,
)

app = modal.App(f"{APP_PREFIX}-train")

DATA = f"{VOL_PATH}/data"


def _lora():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")


def _load_model_tok(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return model, tok


def _final_metrics(trainer, train_out):
    """Pull plain-float training signals out of the trainer (no torch types)."""
    m = {}
    try:
        for k, v in (train_out.metrics or {}).items():
            m[k] = float(v) if isinstance(v, (int, float)) else v
    except Exception:
        pass
    hist = [h for h in trainer.state.log_history if isinstance(h, dict)]
    for key in ("loss", "reward", "rewards/chosen", "kl", "reward_std",
                "rewards/rejected", "logps/chosen"):
        vals = [h[key] for h in hist if key in h]
        if vals:
            m[f"last_{key.replace('/', '_')}"] = float(vals[-1])
    return m


@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 50)
def train(method: str, model: str, run_id: str, seed: int = 0,
          beta: float = 0.1, lr: float = 5e-5, max_steps: int = 150,
          cpo_alpha: float = 0.05, simpo_gamma: float = 0.5,
          num_generations: int = 4, kl_beta: float = 0.04):
    import os, time, random, torch, numpy as np
    from datasets import load_dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_id = BASE_MODELS[model]
    adapter_dir = f"{VOL_PATH}/runs/{run_id}/adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    t0 = time.time()

    common_args = dict(
        output_dir=adapter_dir, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=lr, max_steps=max_steps,
        logging_steps=10, bf16=True, lr_scheduler_type="cosine",
        warmup_ratio=0.1, save_strategy="no", report_to=[], seed=seed,
    )

    if method == "sft":
        from trl import SFTTrainer, SFTConfig
        ds = load_dataset("json", data_files=f"{DATA}/sft.jsonl", split="train")
        args = SFTConfig(max_seq_length=1024, packing=False,
                         **{**common_args, "learning_rate": lr or 2e-4})
        trainer = SFTTrainer(model=model_id, args=args, train_dataset=ds,
                             processing_class=_load_model_tok(model_id)[1],
                             peft_config=_lora())

    elif method == "dpo":
        from trl import DPOTrainer, DPOConfig
        ds = load_dataset("json", data_files=f"{DATA}/prefs.jsonl", split="train")
        m, tok = _load_model_tok(model_id)
        args = DPOConfig(beta=beta, max_length=1024, max_prompt_length=512, **common_args)
        trainer = DPOTrainer(model=m, ref_model=None, args=args, train_dataset=ds,
                             processing_class=tok, peft_config=_lora())

    elif method == "kto":
        from trl import KTOTrainer, KTOConfig
        ds = load_dataset("json", data_files=f"{DATA}/kto.jsonl", split="train")
        m, tok = _load_model_tok(model_id)
        args = KTOConfig(beta=beta, max_length=1024, max_prompt_length=512, **common_args)
        trainer = KTOTrainer(model=m, ref_model=None, args=args, train_dataset=ds,
                             processing_class=tok, peft_config=_lora())

    elif method == "orpo":
        from trl import ORPOTrainer, ORPOConfig
        ds = load_dataset("json", data_files=f"{DATA}/prefs.jsonl", split="train")
        m, tok = _load_model_tok(model_id)
        args = ORPOConfig(beta=beta, max_length=1024, max_prompt_length=512, **common_args)
        trainer = ORPOTrainer(model=m, args=args, train_dataset=ds,
                              processing_class=tok, peft_config=_lora())

    elif method == "simpo":
        from trl import CPOTrainer, CPOConfig
        ds = load_dataset("json", data_files=f"{DATA}/prefs.jsonl", split="train")
        m, tok = _load_model_tok(model_id)
        args = CPOConfig(loss_type="simpo", cpo_alpha=cpo_alpha, beta=max(beta, 2.0),
                         simpo_gamma=simpo_gamma, max_length=1024,
                         max_prompt_length=512, **common_args)
        trainer = CPOTrainer(model=m, args=args, train_dataset=ds,
                             processing_class=tok, peft_config=_lora())

    elif method == "grpo":
        from trl import GRPOTrainer, GRPOConfig
        ds = load_dataset("json", data_files=f"{DATA}/grpo.jsonl", split="train")

        def reward_fn(prompts, completions, target_keyphrases=None, **kw):
            out = []
            for i, comp in enumerate(completions):
                text = comp[-1]["content"] if isinstance(comp, list) else comp
                kps = target_keyphrases[i] if target_keyphrases else []
                r, _ = verifiable_reward(text, kps)
                out.append(r)
            return out

        gargs = dict(common_args)
        gargs.update(per_device_train_batch_size=num_generations,
                     gradient_accumulation_steps=2, remove_unused_columns=False)
        args = GRPOConfig(num_generations=num_generations, max_completion_length=160,
                          max_prompt_length=320, temperature=0.9, beta=kl_beta, **gargs)
        trainer = GRPOTrainer(model=model_id, reward_funcs=reward_fn, args=args,
                              train_dataset=ds, peft_config=_lora())
    else:
        raise ValueError(f"unknown method {method}")

    train_out = trainer.train()
    trainer.save_model(adapter_dir)
    # ensure tokenizer is alongside the adapter for eval
    try:
        trainer.processing_class.save_pretrained(adapter_dir)
    except Exception:
        _load_model_tok(model_id)[1].save_pretrained(adapter_dir)
    VOLUMES[VOL_PATH].commit()

    return {
        "run_id": run_id, "method": method, "model": model, "base_model_id": model_id,
        "seed": seed, "beta": beta, "lr": lr, "max_steps": max_steps,
        "train_runtime_s": round(time.time() - t0, 1),
        "adapter_dir": adapter_dir, "metrics": _final_metrics(trainer, train_out),
    }


@app.local_entrypoint()
def main(method: str = "dpo", model: str = "qwen0.5b", run_id: str = "smoke_dpo",
         seed: int = 0, beta: float = 0.1, lr: float = 5e-5, max_steps: int = 150):
    import os
    # GRPO does online generation each step -> use the fastest GPU so a single
    # train call stays well under the workflow's 10-min per-Bash-call budget.
    fn = train.with_options(gpu="H100") if method == "grpo" else train
    out = fn.remote(method=method, model=model, run_id=run_id, seed=seed,
                    beta=beta, lr=lr, max_steps=max_steps)
    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, f"train_{run_id}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== TRAIN RESULT ===")
    print(json.dumps(out, indent=2))


@app.local_entrypoint()
def smoke(model: str = "qwen0.5b"):
    """Validate that ALL methods instantiate + train + save (3 steps each, parallel)."""
    methods = ["sft", "dpo", "kto", "orpo", "simpo", "grpo"]
    calls = {m: train.spawn(method=m, model=model, run_id=f"smoke_{m}", max_steps=3)
             for m in methods}
    print("launched", list(calls))
    results = {}
    for m, c in calls.items():
        try:
            r = c.get()
            results[m] = {"ok": True, "runtime_s": r["train_runtime_s"],
                          "metrics": r["metrics"]}
        except Exception as e:
            results[m] = {"ok": False, "error": str(e)[:400]}
        print(m, "->", json.dumps(results[m])[:500])
    ok = [m for m, r in results.items() if r["ok"]]
    print(f"\n=== SMOKE: {len(ok)}/{len(methods)} methods OK: {ok} ===")
