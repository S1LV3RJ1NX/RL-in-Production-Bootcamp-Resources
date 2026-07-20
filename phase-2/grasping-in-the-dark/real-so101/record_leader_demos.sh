#!/bin/zsh
# =============================================================================
# Record SO-101 demos for HIL-SERL (option 2): leader teleop via stock lerobot-record
# Task: pick up the duster and wipe the marker off the whiteboard.
#
# Recorded at 10 fps to match the HIL-SERL online env control rate, so the
# offline demos and online RL share the same dynamics. Joint-space actions.
# Saved locally only (no Hub push); the RL learner reads it from the local cache.
#
# Controls (familiar lerobot-record):
#   Right arrow = save episode + next   |   Left arrow = re-record   |   Esc = stop
#
# Usage:
#   ./hilserl_wipe/record_leader_demos.sh
# =============================================================================

export HF_TOKEN="hf_YOUR_TOKEN_HERE"   # ⚠️ rotate after class

source "$(dirname "$0")/../activate.sh"

DATASET_REPO_ID="RajatDandekar/so101_whiteboard_wipe"
NUM_EPISODES=10

# Current calibrated pair
FOLLOWER_PORT="/dev/tty.wchusbserial5AE60830811"
FOLLOWER_ID="my_follower"
LEADER_PORT="/dev/tty.wchusbserial5AE60829961"
LEADER_ID="right_leader"

TASK="Pick up the duster and wipe the marker off the whiteboard"

# Fresh dataset (remove any stale cache with this id)
rm -rf ~/.cache/huggingface/lerobot/${DATASET_REPO_ID}

echo "============================================"
echo "  Task:      $TASK"
echo "  Episodes:  $NUM_EPISODES  @ 10 fps"
echo "  Dataset:   $DATASET_REPO_ID (local + Hub backup)"
echo "  Follower:  $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "  Leader:    $LEADER_PORT ($LEADER_ID)"
echo "  Cameras:   front=0, wrist=1"
echo "============================================"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.push_to_hub=false \
  --display_data=true

echo ""
echo "=== Done recording! Dataset at ~/.cache/huggingface/lerobot/${DATASET_REPO_ID} ==="
