"""Patch LeRobot's image writer so RL replay-buffer/offline-dataset serialization doesn't crash.

Root cause (found by source audit): the SAC replay buffer stores image obs as float32 with raw
[0,255] pixel values (buffer.py allocates float32, never casts). At each checkpoint the learner
dumps the buffer via `to_lerobot_dataset`, and `image_array_to_pil_image` RAISES on float images
whose values exceed 1.0 — killing the training loop at the first save.

Fix: accept float images already in [0,255] by clip+cast to uint8 (true [0,1] floats keep the
original *255 scaling). This is training-neutral — it only touches on-disk dataset serialization,
never the normalized batch the policy trains on (normalization is applied to the sampled batch in
the trainer, not to buffer storage).

Applied at Modal image-build time against the editable install at /root/lerobot.
Idempotent: safe to run more than once.
"""
import sys

PATH = "/root/lerobot/src/lerobot/datasets/image_writer.py"

OLD = '''    if image_array.dtype != np.uint8:
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "The image data type is float, which requires values in the range [0.0, 1.0]. "
                    f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                    "provide a uint8 image with values in the range [0, 255]."
                )

        image_array = (image_array * 255).astype(np.uint8)'''

NEW = '''    if image_array.dtype != np.uint8:
        # [vizuara patch] Accept float images already in [0, 255] pixel range (RL replay-buffer
        # dumps store raw gym-hil uint8 obs as float32) by clip+cast to uint8 instead of raising.
        # True [0, 1] floats keep the original *255 scaling. Training-neutral: affects only the
        # on-disk dataset serialization, never the sampled batch the policy trains on.
        max_ = image_array.max().item()
        min_ = image_array.min().item()
        if max_ > 1.0 or min_ < 0.0:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        else:
            image_array = (image_array * 255).astype(np.uint8)'''

MARKER = "[vizuara patch]"


def main() -> int:
    with open(PATH) as f:
        src = f.read()
    if MARKER in src:
        print("[patch] image_writer.py already patched — skipping")
        return 0
    if OLD not in src:
        print("[patch] ERROR: target block not found in", PATH)
        print("[patch] LeRobot's image_writer.py changed; update OLD/NEW in this script.")
        return 1
    with open(PATH, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    print("[patch] image_writer.py patched OK ->", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
