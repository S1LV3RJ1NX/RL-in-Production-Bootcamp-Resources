"""
Capture the follower's current joint pose (calibrated degrees) to use as
`fixed_reset_joint_positions` in the HIL-SERL config.

Usage:
    1. Move the follower (via the leader) to a good neutral start pose:
       arm up, gripper open, clear of the whiteboard.
    2. cd /Users/rajatdandekar/Desktop/Robotics/lerobot && source .venv/bin/activate
    3. python /Users/rajatdandekar/Desktop/Robotics/hilserl_wipe/capture_reset_pose.py
    4. Copy the printed list into record_config.json ->
       env.processor.reset.fixed_reset_joint_positions
"""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/tty.wchusbserial5AE60830811"
FOLLOWER_ID = "my_follower"

ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)  # loads existing my_follower calibration
    try:
        obs = robot.get_observation()
        pose = [round(float(obs[f"{m}.pos"]), 1) for m in ORDER]
        print("\n" + "=" * 60)
        print("RESET POSE (calibrated degrees), in motor order:")
        print(ORDER)
        print(pose)
        print("\nPaste this into record_config.json:")
        print(f'  "fixed_reset_joint_positions": {pose}')
        print("=" * 60 + "\n")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
