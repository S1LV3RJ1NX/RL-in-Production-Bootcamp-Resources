#!/bin/zsh
# =============================================================================
# RESUME recording HIL-SERL demos (appends to existing dataset, deletes nothing).
# Use this after a disconnect or to add more episodes to so101_whiteboard_wipe.
#
# Records ADDITIONAL episodes (num_episodes = new episodes this session).
# Re-run safely as many times as needed — it always appends.
#
# Usage:
#   ./hilserl_wipe/resume_leader_demos.sh [num_new_episodes]   # default 7
# =============================================================================

source "$(dirname "$0")/../activate.sh"

DATASET_REPO_ID="RajatDandekar/so101_whiteboard_wipe"
NUM_NEW="${1:-7}"

FOLLOWER_PORT="/dev/tty.wchusbserial5AE60830811"
FOLLOWER_ID="my_follower"
LEADER_PORT="/dev/tty.wchusbserial5AE60829961"
LEADER_ID="right_leader"
TASK="Pick up the duster and wipe the marker off the whiteboard"

echo "Resuming $DATASET_REPO_ID — recording $NUM_NEW more episode(s), appending."

lerobot-record \
  --robot.type=so101_follower --robot.port="${FOLLOWER_PORT}" --robot.id="${FOLLOWER_ID}" \
  --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader --teleop.port="${LEADER_PORT}" --teleop.id="${LEADER_ID}" \
  --dataset.repo_id="${DATASET_REPO_ID}" --dataset.single_task="${TASK}" \
  --dataset.fps=10 --dataset.episode_time_s=120 --dataset.reset_time_s=15 \
  --dataset.num_episodes="${NUM_NEW}" --dataset.push_to_hub=false --resume=true

echo ""
echo "=== Done. Re-run this script to add more if needed. ==="
