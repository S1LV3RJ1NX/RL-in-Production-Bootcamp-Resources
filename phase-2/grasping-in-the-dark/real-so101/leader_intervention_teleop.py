"""
Custom SO-101 leader teleoperator for JOINT-SPACE HIL-SERL interventions.

Why this exists: stock lerobot has no way to use the leader arm for HIL-SERL
interventions. This teleop makes the leader work by:
  1. Returning the leader's 6 joint positions as a NORMALIZED numpy array in [-1, 1]
     (the same space the SAC policy outputs), so the stock InterventionActionProcessor
     (which accepts a raw np.ndarray action) feeds them straight to the follower.
  2. Providing get_teleop_events() via a keyboard listener:
       - SPACE  = toggle intervention on/off (while on, the leader drives the follower)
       - s      = mark success (reward = 1, ends episode)
       - r      = end + rerecord
       - q      = end as failure

Action-space convention (must match run_actor.py's unnormalize + the offline demos):
  normalized = 2 * (deg - MIN) / (MAX - MIN) - 1   ->  [-1, 1]
  deg        = (normalized + 1) / 2 * (MAX - MIN) + MIN
MIN/MAX are the per-joint action ranges from the recorded dataset's stats.

Registered as teleop type "so101_leader_intervention". Import this module before
parsing a config that references it (run_actor.py / run_learner.py do this).
"""

from dataclasses import dataclass
from queue import Queue

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.utils import TeleopEvents

# Per-joint action ranges (degrees; gripper 0-100), from the recorded dataset stats.
MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ACTION_MIN = np.array([-44.57, -86.68, -105.41, 28.31, -27.03, 0.0], dtype=np.float32)
ACTION_MAX = np.array([93.27, 98.9, 81.49, 107.96, 35.12, 97.41], dtype=np.float32)
_RANGE = ACTION_MAX - ACTION_MIN


def normalize_action(deg: np.ndarray) -> np.ndarray:
    """Joint degrees -> normalized [-1, 1]."""
    a = 2.0 * (np.asarray(deg, dtype=np.float32) - ACTION_MIN) / _RANGE - 1.0
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def unnormalize_action(a) -> np.ndarray:
    """Normalized [-1, 1] -> joint degrees. Accepts torch tensors or numpy arrays."""
    if hasattr(a, "detach"):  # torch tensor
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    return ((a + 1.0) / 2.0 * _RANGE + ACTION_MIN).astype(np.float32)


@TeleoperatorConfig.register_subclass("so101_leader_intervention")
@dataclass
class SOLeaderInterventionConfig(SOLeaderTeleopConfig):
    pass


class SOLeaderIntervention(SOLeader):
    """SO leader that outputs normalized joint actions + keyboard intervention events."""

    config_class = SOLeaderInterventionConfig
    # NOTE: keep name "so_leader" so it reuses the existing calibration at
    # calibration/teleoperators/so_leader/<id>.json (right_leader) instead of
    # prompting for a fresh calibration. The registry type is still
    # "so101_leader_intervention" (set via register_subclass), which is what the
    # config references; `name` only drives the calibration dir + logging repr.
    name = "so_leader"

    def __init__(self, config: SOLeaderInterventionConfig):
        super().__init__(config)
        self._intervention = False
        self._misc_queue: Queue = Queue()
        self._listener = None

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("[leader-intervention] pynput not available; events disabled.")
            return

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self._intervention = not self._intervention
                    print(f"[leader-intervention] intervention {'ON' if self._intervention else 'OFF'}")
                elif hasattr(key, "char") and key.char in ("s", "r", "q"):
                    self._misc_queue.put(key.char)
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        print("[leader-intervention] keyboard listener started (SPACE=toggle intervene, s/r/q).")

    def get_action(self) -> np.ndarray:
        """Return leader joints normalized to [-1, 1] as a numpy array (motor order)."""
        d = super().get_action()  # {"<motor>.pos": value}
        deg = np.array([d[f"{m}.pos"] for m in MOTOR_ORDER], dtype=np.float32)
        return normalize_action(deg)

    def get_teleop_events(self) -> dict:
        success = False
        rerecord = False
        terminate = False
        while not self._misc_queue.empty():
            k = self._misc_queue.get_nowait()
            if k == "s":
                success = True
            elif k == "r":
                terminate = True
                rerecord = True
            elif k == "q":
                terminate = True
        return {
            TeleopEvents.IS_INTERVENTION: self._intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord,
        }

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        super().disconnect()
