"""
Add a sparse success reward to the cropped demo dataset so it can seed the SAC
offline replay buffer.

Why: `lerobot-record` does not store rewards, but lerobot's RL offline buffer
(`ReplayBuffer.from_lerobot_dataset`) requires a `next.reward` column. These are
all successful expert demos, so we set reward = 1.0 on the LAST frame of each
episode (task completed) and 0.0 elsewhere — a sparse success signal that matches
the online reward you'll give by pressing `s`.

One-time data-prep. Operates on the (regenerable) *_cropped_resized dataset.

Usage:
    python hilserl_wipe/add_reward.py
"""

import json
import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CDS = os.path.expanduser(
    "~/.cache/huggingface/lerobot/RajatDandekar/so101_whiteboard_wipe_cropped_resized"
)
REWARD_KEY = "next.reward"


def main():
    # --- 1) parquet: add next.reward = 1.0 on each episode's last frame ---
    files = sorted(glob.glob(os.path.join(CDS, "data", "**", "*.parquet"), recursive=True))
    total_success = 0
    for f in files:
        df = pd.read_parquet(f)
        if REWARD_KEY in df.columns:
            print(f"  {os.path.basename(f)}: already has {REWARD_KEY}, overwriting")
        df[REWARD_KEY] = 0.0
        # last row index per episode (rows are ordered within the file)
        last_rows = df.groupby("episode_index").tail(1).index
        df.loc[last_rows, REWARD_KEY] = 1.0
        df[REWARD_KEY] = df[REWARD_KEY].astype("float32")
        total_success += len(last_rows)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)
        print(f"  {os.path.basename(f)}: {len(df)} rows, {len(last_rows)} success frames")

    # --- 2) meta/info.json: register the feature ---
    info_path = os.path.join(CDS, "meta", "info.json")
    info = json.load(open(info_path))
    info["features"][REWARD_KEY] = {"dtype": "float32", "shape": [1], "names": None}
    json.dump(info, open(info_path, "w"), indent=4)

    print(f"\nDone. Added {REWARD_KEY} ({total_success} success frames total) + registered in info.json")


if __name__ == "__main__":
    main()
