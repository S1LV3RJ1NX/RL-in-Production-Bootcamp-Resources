"""Classic RLHF: train a reward model on the Socratic preferences, then PPO the
policy against it. Kept separate from train.py because it needs 4 models in
memory (policy/ref/reward/value) and is the most finicky recipe.

Run RM:   modal run modal_apps/ppo.py::smoke_rm
Run PPO:  modal run modal_apps/ppo.py --model qwen0.5b --run-id ppo_qwen0.5b_s0
"""
import json
import modal
from common import (
    APP_PREFIX, TRAIN_IMAGE, VOLUMES, VOL_PATH, BASE_MODELS, GPU_SMALL, TUTOR_SYSTEM,
)

app = modal.App(f"{APP_PREFIX}-ppo")
DATA = f"{VOL_PATH}/data"


def _lora_seqcls():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="SEQ_CLS", target_modules="all-linear",
                      modules_to_save=["score"])


@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 40)
def train_rm(model: str, run_id: str, max_steps: int = 200, lr: float = 1e-5,
             seed: int = 0):
    """Bradley-Terry reward model on the preference pairs; merged + saved."""
    import os, time, torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardTrainer, RewardConfig
    from datasets import load_dataset
    from peft import PeftModel

    model_id = BASE_MODELS[model]
    rm_dir = f"{VOL_PATH}/runs/{run_id}_rm"
    adapter_dir = f"{rm_dir}/adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=f"{DATA}/prefs.jsonl", split="train")

    def to_rm(ex):  # full conversation = prompt + completion, conversational
        return {"chosen": ex["prompt"] + ex["chosen"],
                "rejected": ex["prompt"] + ex["rejected"]}

    ds = ds.map(to_rm, remove_columns=ds.column_names)

    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, torch_dtype=torch.bfloat16)
    mdl.config.pad_token_id = tok.pad_token_id

    args = RewardConfig(
        output_dir=adapter_dir, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=lr, max_steps=max_steps,
        max_length=1024, bf16=True, save_strategy="no", report_to=[],
        logging_steps=10, seed=seed, center_rewards_coefficient=0.01,
        remove_unused_columns=False)
    trainer = RewardTrainer(model=mdl, args=args, train_dataset=ds,
                            processing_class=tok, peft_config=_lora_seqcls())
    out = trainer.train()
    trainer.save_model(adapter_dir)  # persist the LoRA adapter first
    # merge LoRA into a standalone RM so PPO can load it directly
    merged = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=1, torch_dtype=torch.bfloat16),
        adapter_dir).merge_and_unload()
    merged.config.pad_token_id = tok.pad_token_id
    merged.save_pretrained(rm_dir)
    tok.save_pretrained(rm_dir)
    VOLUMES[VOL_PATH].commit()
    m = out.metrics or {}
    return {"run_id": run_id, "rm_dir": rm_dir, "model": model,
            "train_runtime_s": round(time.time() - t0, 1),
            "train_loss": float(m.get("train_loss", 0.0))}


@app.function(image=TRAIN_IMAGE, gpu="H100", volumes=VOLUMES, timeout=60 * 50)
def train_ppo(model: str, run_id: str, rm_run_id: str = "", total_episodes: int = 2000,
              lr: float = 1e-6, seed: int = 0, kl_coef: float = 0.05):
    """PPO the policy against the trained reward model (verifiable-reward-free,
    classic RLHF). Saves a LoRA-free merged policy adapter dir for eval."""
    import os, time, torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForSequenceClassification)
    from trl import PPOTrainer, PPOConfig
    from datasets import load_dataset

    model_id = BASE_MODELS[model]
    rm_run_id = rm_run_id or run_id
    rm_dir = f"{VOL_PATH}/runs/{rm_run_id}_rm"
    out_dir = f"{VOL_PATH}/runs/{run_id}/adapter"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # prompt dataset -> tokenized "input_ids" (chat template, generation prompt)
    ds = load_dataset("json", data_files=f"{DATA}/grpo.jsonl", split="train")

    def tok_prompt(ex):
        text = tok.apply_chat_template(ex["prompt"], tokenize=False,
                                       add_generation_prompt=True)
        return {"input_ids": tok(text, truncation=True, max_length=320)["input_ids"]}

    ds = ds.map(tok_prompt, remove_columns=ds.column_names)

    policy = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    ref = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_dir, num_labels=1, torch_dtype=torch.bfloat16)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, torch_dtype=torch.bfloat16)
    for m in (reward_model, value_model):
        m.config.pad_token_id = tok.pad_token_id

    args = PPOConfig(
        output_dir=out_dir, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_mini_batches=1, num_ppo_epochs=2,
        total_episodes=total_episodes, learning_rate=lr, response_length=200,
        kl_coef=kl_coef, temperature=0.9, bf16=True, report_to=[],
        logging_steps=5, save_strategy="no", seed=seed,
        missing_eos_penalty=1.0, num_sample_generations=0)
    trainer = PPOTrainer(args=args, processing_class=tok, model=policy,
                         ref_model=ref, reward_model=reward_model,
                         value_model=value_model, train_dataset=ds)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    VOLUMES[VOL_PATH].commit()
    return {"run_id": run_id, "method": "ppo", "model": model,
            "adapter_dir": out_dir, "rm_dir": rm_dir,
            "train_runtime_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def smoke_rm(model: str = "qwen0.5b"):
    print(json.dumps(train_rm.remote(model=model, run_id="smoke", max_steps=5), indent=2))


@app.local_entrypoint()
def smoke(model: str = "qwen0.5b"):
    """Validate the full RM->PPO path cheaply (RM 5 steps, PPO ~64 episodes)."""
    rm = train_rm.remote(model=model, run_id="ppo_smoke", max_steps=5)
    print("RM:", json.dumps(rm))
    try:
        ppo = train_ppo.remote(model=model, run_id="ppo_smoke", rm_run_id="ppo_smoke",
                               total_episodes=64)
        print("PPO OK:", json.dumps(ppo))
    except Exception as e:
        print("PPO FAIL:", str(e)[:600])


@app.local_entrypoint()
def main(model: str = "qwen0.5b", run_id: str = "ppo_qwen0.5b_s0",
         total_episodes: int = 2000):
    rm = train_rm.remote(model=model, run_id=run_id, max_steps=200)
    print("RM:", json.dumps(rm))
    ppo = train_ppo.remote(model=model, run_id=run_id, rm_run_id=run_id,
                           total_episodes=total_episodes)
    print("PPO:", json.dumps(ppo))
