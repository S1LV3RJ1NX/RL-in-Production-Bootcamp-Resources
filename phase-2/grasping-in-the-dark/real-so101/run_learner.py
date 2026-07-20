"""
Minimal launcher for the stock LeRobot SAC learner.

Why this exists: lerobot's learner (`python -m lerobot.rl.learner`) imports only the
gamepad and so_leader teleoperators, so it cannot PARSE a config whose
`env.teleop.type` is `keyboard_ee` (the keyboard module never gets imported, so that
choice isn't registered). The actor is unaffected because it imports gym_manipulator,
which imports the keyboard module.

This shim adds that one missing import and then hands off to the *stock* learner
entry point unchanged. It contains no logic of its own.

Usage:
    python hilserl_wipe/run_learner.py --config_path hilserl_wipe/train_config.json
"""

import lerobot.teleoperators.keyboard  # noqa: F401  # registers "keyboard"/"keyboard_ee"
import leader_intervention_teleop  # noqa: F401  # registers "so101_leader_intervention"
from lerobot.rl.learner import train_cli

if __name__ == "__main__":
    train_cli()
