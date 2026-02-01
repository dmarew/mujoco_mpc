# G1 Motion Tracking Task

Motion tracking task for the Unitree G1 humanoid robot using CSV reference trajectories.

## CSV Format

```
root_x, root_y, root_z, quat_w, quat_x, quat_y, quat_z, joint_0, joint_1, ..., joint_N
```

Each row is one frame:
- **Columns 0-2**: Root position (x, y, z) in meters
- **Columns 3-6**: Root orientation quaternion (w, x, y, z)
- **Columns 7+**: Joint positions in radians (model qpos order, excluding floating base)

The joint order follows the model's natural `qpos` order - no need to specify names.

## Example

```csv
0.0,0.0,0.755,1.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

## Usage

1. Generate a trajectory:
```bash
python3 generate_example_trajectory.py --motion squat --output my_motion.csv --fps 30 --duration 3.0
```

2. Add to task.xml `<custom>` section:
```xml
<text name="trajectory_path" data="my_motion.csv" />
<numeric name="trajectory_fps" data="30" />
```

3. Run:
```bash
./bin/mjpc --task="G1 Track"
```

## Available Motions

- `stand` - Static standing
- `squat` - Squatting motion
- `walk` - Walking in place
- `wave` - Arm waving
