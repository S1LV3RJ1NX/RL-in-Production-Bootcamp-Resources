"""
Normalize the offline demo dataset's `action` column from joint degrees to [-1, 1],
so the offline replay buffer matches the online action space (the SAC policy outputs
[-1, 1], and run_actor.py unnormalizes to degrees only at the robot). Without this,
the learner would mix degree-scale (offline) and [-1,1]-scale (online) actions.

Uses the SAME ACTION_MIN/MAX as leader_intervention_teleop, so all three action
sources (offline demos, online policy, leader interventions) live in one space.

Idempotent guard: skips if the action column already looks normalized ([-1,1]).
One-time. Run after add_reward.py, before (re)starting the learner.

    python hilserl_wipe/normalize_offline_actions.py
"""

import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from leader_intervention_teleop import ACTION_MIN, ACTION_MAX, normalize_action

CDS = os.path.expanduser(
    "~/.cache/huggingface/lerobot/RajatDandekar/so101_whiteboard_wipe_cropped_resized"
)


def main():
    files = sorted(glob.glob(os.path.join(CDS, "data", "**", "*.parquet"), recursive=True))
    for f in files:
        df = pd.read_parquet(f)
        actions = np.stack(df["action"].to_numpy())  # (N, 6) in degrees
        amin, amax = actions.min(axis=0), actions.max(axis=0)
        # Guard: if already within [-1.01, 1.01], assume normalized and skip.
        if amin.min() >= -1.01 and amax.max() <= 1.01:
            print(f"  {os.path.basename(f)}: already normalized, skipping")
            continue
        print(f"  {os.path.basename(f)}: action range before  min={np.round(amin,1)} max={np.round(amax,1)}")
        norm = np.stack([normalize_action(a) for a in actions]).astype(np.float32)
        df["action"] = list(norm)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)
        after = np.stack(df["action"].to_numpy())
        print(f"  {os.path.basename(f)}: action range after   min={np.round(after.min(axis=0),2)} max={np.round(after.max(axis=0),2)}")

    print(f"\nDone. Offline actions normalized to [-1, 1] using MIN={ACTION_MIN.tolist()} MAX={ACTION_MAX.tolist()}")


if __name__ == "__main__":
    main()
