"""Force-build both images and assert every library + TRL trainer we rely on
imports cleanly on a GPU. Run: modal run modal_apps/build_check.py
"""
import modal
from common import APP_PREFIX, INFER_IMAGE, TRAIN_IMAGE, GPU_SMALL

app = modal.App(f"{APP_PREFIX}-build-check")


@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, timeout=600)
def check_train():
    import torch, transformers, trl, peft, accelerate, datasets
    # the exact trainers the sweep needs
    from trl import (
        SFTTrainer, SFTConfig,
        DPOTrainer, DPOConfig,
        KTOTrainer, KTOConfig,
        ORPOTrainer, ORPOConfig,
        CPOTrainer, CPOConfig,        # SimPO == CPO(loss_type="simpo")
        GRPOTrainer, GRPOConfig,
        PPOTrainer, PPOConfig,
        RewardTrainer, RewardConfig,
    )
    import lm_eval
    from importlib.metadata import version as _ver
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "lm_eval": _ver("lm_eval"),
        "trainers_ok": True,
    }
    print("TRAIN_IMAGE OK:", info)
    return info


@app.function(image=INFER_IMAGE, gpu=GPU_SMALL, timeout=600)
def check_infer():
    import json, traceback
    try:
        import vllm, transformers, torch
        info = {
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        print("INFER_IMAGE OK:", info)
        return json.dumps(info)
    except Exception:
        tb = traceback.format_exc()
        print("INFER_IMAGE FAILED:\n", tb)
        return json.dumps({"error": tb[-1500:]})


@app.local_entrypoint()
def main():
    i = check_infer.remote()
    print("\n=== INFER RESULT ===")
    print(i)
