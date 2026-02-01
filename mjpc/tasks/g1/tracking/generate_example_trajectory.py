#!/usr/bin/env python3
"""Generate example motion trajectories for G1 motion tracking task.

Usage:
    python generate_example_trajectory.py --motion squat --output squat.csv --fps 30 --duration 3.0
"""

import argparse
import csv
import math

# Number of joints in G1 model (excluding floating base)
NUM_JOINTS = 23

# Joint indices (for reference, matches model qpos order):
# 0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
# 6-11: right leg (same order)
# 12: torso
# 13-17: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll)
# 18-22: right arm (same order)


def generate_stand(duration: float, fps: float) -> list:
    """Generate standing pose trajectory."""
    num_frames = int(duration * fps)
    # Format: root_pos(3), quat_xyzw(4), joints
    frame = [0.0, 0.0, 0.755, 0.0, 0.0, 0.0, 1.0] + [0.0] * NUM_JOINTS
    return [frame for _ in range(num_frames)]


def generate_squat(duration: float, fps: float) -> list:
    """Generate squatting motion trajectory."""
    num_frames = int(duration * fps)
    frames = []

    for i in range(num_frames):
        t = i / fps
        phase = 0.5 * (1 - math.cos(2 * math.pi * t / duration))

        squat_depth = phase * 0.3
        knee_angle = phase * 1.0
        hip_angle = phase * 0.5

        joints = [0.0] * NUM_JOINTS
        # Left leg
        joints[0] = -hip_angle      # left_hip_pitch
        joints[3] = knee_angle      # left_knee
        joints[4] = hip_angle * 0.5 # left_ankle_pitch
        # Right leg
        joints[6] = -hip_angle      # right_hip_pitch
        joints[9] = knee_angle      # right_knee
        joints[10] = hip_angle * 0.5 # right_ankle_pitch

        frame = [0.0, 0.0, 0.755 - squat_depth, 0.0, 0.0, 0.0, 1.0] + joints
        frames.append(frame)

    return frames


def generate_walk_in_place(duration: float, fps: float) -> list:
    """Generate walking in place motion trajectory."""
    num_frames = int(duration * fps)
    frames = []

    for i in range(num_frames):
        t = i / fps
        phase = 2 * math.pi * t

        left_lift = max(0, math.sin(phase))
        right_lift = max(0, math.sin(phase + math.pi))
        root_z = 0.755 + 0.02 * math.cos(2 * phase)

        joints = [0.0] * NUM_JOINTS
        # Left leg
        joints[0] = -left_lift * 0.3   # hip_pitch
        joints[3] = left_lift * 0.6    # knee
        # Right leg
        joints[6] = -right_lift * 0.3  # hip_pitch
        joints[9] = right_lift * 0.6   # knee
        # Arm swing
        joints[13] = right_lift * 0.2  # left_shoulder_pitch
        joints[18] = left_lift * 0.2   # right_shoulder_pitch

        frame = [0.0, 0.0, root_z, 0.0, 0.0, 0.0, 1.0] + joints
        frames.append(frame)

    return frames


def generate_arm_wave(duration: float, fps: float) -> list:
    """Generate arm waving motion trajectory."""
    num_frames = int(duration * fps)
    frames = []

    for i in range(num_frames):
        t = i / fps
        wave = 2 * math.pi * t * 2

        joints = [0.0] * NUM_JOINTS
        joints[18] = -1.0                          # right_shoulder_pitch
        joints[19] = -0.5 + 0.3 * math.sin(wave)   # right_shoulder_roll
        joints[21] = 0.5 + 0.3 * math.sin(wave)    # right_elbow_pitch

        frame = [0.0, 0.0, 0.755, 0.0, 0.0, 0.0, 1.0] + joints
        frames.append(frame)

    return frames


def write_csv(frames: list, output_path: str):
    """Write frames to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for frame in frames:
            writer.writerow([f'{v:.6f}' for v in frame])
    print(f"Wrote {len(frames)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate G1 motion trajectories')
    parser.add_argument('--motion', type=str, default='stand',
                        choices=['stand', 'squat', 'walk', 'wave'])
    parser.add_argument('--output', type=str, default='trajectory.csv')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--duration', type=float, default=2.0)
    args = parser.parse_args()

    generators = {
        'stand': generate_stand,
        'squat': generate_squat,
        'walk': generate_walk_in_place,
        'wave': generate_arm_wave,
    }

    frames = generators[args.motion](args.duration, args.fps)
    write_csv(frames, args.output)
    print(f"Generated {args.motion} at {args.fps} FPS for {args.duration}s")


if __name__ == '__main__':
    main()
