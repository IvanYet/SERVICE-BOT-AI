# RoboCup@Home Robot Software

The current Receptionist1 prototype uses speech, an Ollama-based language
model for name confirmation, Nav2 navigation, YOLO/BoT-SORT person tracking,
and OSNet appearance re-identification. After “Hey Jarvis,” it navigates to the
door, confirms the guest's name, enrolls a multi-view appearance gallery under
a logical `guest_N` ID, then guides the guest through the task. It keeps each
guest's conversation details associated with that logical ID.

## Layout

```text
robo_cup_robot/
├── speech/           # Wake word, recording, transcription, LLM, speech output
├── vision/           # YOLO/RealSense detection and Qwen3.5 appearance features
├── navigation/       # ROS 2 simulation workspace, map, world, and Nav2 config
├── task_manager/     # Application entry point and task orchestration
└── task_1/            # Receptionist task
```

## Navigation simulation

The navigation workspace lives at `navigation/ros2_ws/`. It contains all five
ROS 2 source packages from the separate `nav_ws/src` workspace:

- `robot_gazebo`: simulation world, saved map, and simulation launch files.
- `my_robot_description`: robot URDF/Xacro, including the Gazebo sensor and
  drive plugins.
- `my_robot_bringup`: robot and sensor launch files, RViz config, EKF and Nav2
  parameters.
- `my_esp_driver`: ESP32 bridge, online mapping config, and real robot Nav2
  launch/config.
- `sllidar_ros2`: RPLIDAR ROS 2 driver source.

The authoritative 2D map is the LiDAR-created
`my_esp_driver/maps/my_room_map.pgm`, with its required metadata in
`my_esp_driver/maps/my_room_map.yaml` (5 cm per pixel). It shows the room and
the two rectangular obstacles. The separate simplified `sim_map` was removed
because it does not represent your scanned room. The source workspace's
generated `build`, `install`, and `log` directories were not copied; build the
integrated workspace locally.

Build the nested ROS 2 workspace once:

```bash
cd /home/eyec/Desktop/robocupEdu/robo_cup_robot/navigation/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Gazebo is optional. If you use it later, start the imported Gazebo world and
robot in one terminal:

```bash
cd /home/eyec/Desktop/robocupEdu/robo_cup_robot/navigation/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_gazebo gazebo.launch.py
```

Start Nav2 and RViz in another terminal. The launch now loads your LiDAR map:

```bash
cd /home/eyec/Desktop/robocupEdu/robo_cup_robot/navigation/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_gazebo sim_navigation.launch.py
```

The saved LiDAR map can be used on the real robot without Gazebo. For that,
the robot must publish live LiDAR scans, odometry, and the required TF frames,
and its ESP32 bridge must accept Nav2 velocity commands. The imported hardware
bringup does not start the ESP32 bridge yet. Set the initial pose in RViz to
the robot's actual position on the map before sending a goal.

The optional Gazebo world is only a rough room scene; it does not include the
two rectangular obstacles from the scanned map or mapped space outside the
door. Do not use it to validate the doorway start until those are aligned.

Start the RealSense driver in terminal 1 with aligned 640x480 color/depth,
matching the seat-reference workspace:

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30
```

Start the person, pose, chair, and guest-identity vision node in terminal 2
from the project root. The bundled YOLO11n pose weights are at
`robo_cup_robot/vision/models/yolo11n-pose.pt`. Qwen3.5 scene descriptions are
enabled by default; set
`RECEPTIONIST_VLM=0` to disable them:

```bash
cd /home/eyec/Desktop/robocupEdu
source /opt/ros/$ROS_DISTRO/setup.bash
export RECEPTIONIST_VLM_MODEL=qwen3.5:2b
python3 -m robo_cup_robot.vision.receptionist_vision_node
```

Start the speech and task manager in terminal 3 from the project root, with
ROS sourced so its identity and navigation monitor can connect:

```bash
cd /home/eyec/Desktop/robocupEdu
source /opt/ros/$ROS_DISTRO/setup.bash
python3 -m robo_cup_robot.task_manager.main
```

Start Nav2 and RViz in terminal 4 from the navigation workspace:

```bash
cd /home/eyec/Desktop/robocupEdu/robo_cup_robot/navigation/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch my_esp_driver navigation.launch.py
```

This launches map-based Nav2 and RViz. On the physical robot it also needs
live LiDAR, odometry, and TF from robot bringup, plus the ESP32 bridge to carry
`/cmd_vel` to the motors. The launch does not start those components itself.
Receptionist1 sends real Nav2 movement goals to the configured door and table
poses. It does not publish `/initialpose`; AMCL must already be localized at
the robot's actual current position before the task starts.

The vision node publishes chair/couch seat status and person snapshots on
`/robot/vision/receptionist_data`. It uses the standard `yolov8n.pt` COCO
classes for people (0), chairs (56), and couches (57); the drink detector's
`best.pt` is separate and is not used for seat boxes. Seat boxes must score at
least 0.75 and pass the seat-reference workspace's size and aspect-ratio
filters. A separate seat tracker keeps stable seat IDs across temporary
detector-ID changes, requires one second of evidence before changing a seat to
empty or occupied, and retains occupied status through brief occlusions. It
uses YOLO11n-pose when available, with a conservative person-box/seat-overlap
fallback when pose evidence is unavailable. Ambiguous poses and nested/stacked
chairs are marked uncertain and are not offered; standing people do not mark a
place occupied. The task manager
navigates to the seating-area waypoint, selects the nearest empty seat with a
valid depth reading, tells the guest its camera-relative left/center/right
position, and waits for its occupancy status to change. A manual one-frame scene description can be requested with
`/robot/vision/describe_request`; the reply arrives on
`/robot/vision/description`. Qwen3.5 is optional for scene descriptions. When
enabled, it uses CPU inference with an eight-thread maximum and a 120-second
timeout (`RECEPTIONIST_VLM_TIMEOUT` can change the timeout).

When the vision node is running, its loopback-only tuning page is available at
`http://127.0.0.1:8765`. It shows the annotated camera stream and current ROS
vision output, and lets you change seat detection/occupancy and OSNet ReID
thresholds live. Changes are saved to
`/home/eyec/Desktop/robocupEdu/receptionist_vision_tuning.json`; delete that
file to restore defaults. To use another port, set
`RECEPTIONIST_TUNING_PORT` before starting the vision node.

## Guest identity

YOLO's BoT-SORT tracker supplies temporary person IDs, and the bundled OSNet
model compares person appearance against a frozen per-guest gallery. A
confirmed guest keeps the same logical `guest_N` ID when their tracker ID
changes. Guest identity remains independent of chair tracking. Chair occupancy uses
pose keypoints and chair/person image overlap; it does not assign a logical
guest identity to a chair.

During Receptionist1, after the guest confirms their name, the robot asks them
to stand clearly in view and turn slowly. It captures ten appearance samples
before guiding them onward. A previously enrolled guest may remain visible;
tracks already bound to enrolled guests when enrollment starts are excluded.
If several new people are visible, enrollment selects the nearest one using
aligned depth and keeps that track fixed while it remains visible. If depth is
missing for any of multiple new candidates, enrollment waits instead of
guessing. Unresolved identities still block enrollment. The gallery stays in
memory for that vision-node run; restarting the node requires guests to be
enrolled again. Appearance ReID is not proof of a person's real-world name.

The OSNet model and encoder use CPU ONNX Runtime. Install it in the same Python
environment used to launch the vision node if it is missing:

```bash
python3 -m pip install onnxruntime
```

Optional initial matching settings can be adjusted with
`RECEPTIONIST_REID_THRESHOLD` (default `0.60`),
`RECEPTIONIST_TRACK_THRESHOLD` (default `0.60`),
`RECEPTIONIST_REID_MARGIN` (default `0.06`), and
`RECEPTIONIST_REID_HITS` (default `3`). These are starting values, not
Receptionist-camera calibration results. Collect genuine-match and different-
person scores in the actual room before relying on identity recovery in a live
run. Thresholds that leave any doubt should keep the identity ambiguous.

Run the offline identity association checks from the workspace root:

```bash
cd /home/eyec/Desktop/robocupEdu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  robo_cup_robot/vision/test_guest_identity.py \
  robo_cup_robot/vision/test_receptionist_monitor_identity.py \
  robo_cup_robot/task_manager/test_guest_record.py
```

The added person fields are published on the existing
`/robot/vision/receptionist_data` topic. Person entries carry `tracker_id`,
`guest_id`, `identity_state`, and `identity_confidence`. Before introductions,
Task Manager requires both enrolled guests to be visible and uniquely
identified; it stops the introductions when that evidence is missing or
ambiguous. At the fixed seating-area waypoint, the robot offers a visible free
chair or couch and waits for the seat tracker to confirm occupancy. It describes
the selected place by its left/center/right image position; the current robot
code does not actuate a pointing arm.

For a terminal-only snapshot description, run this from the project root while
the RealSense driver is publishing camera images:

```bash
source /opt/ros/humble/setup.bash
python3 -m robo_cup_robot.vision.describe_camera
```

It opens a live color preview immediately, captures the latest frame after five
seconds, prints Qwen3.5's structured feature answers, then prints a paragraph
paraphrased by the language model while keeping the preview live. The default
vision model is `qwen3.5:2b`; the default paragraph model is `qwen2.5:1.5b`.
Each run overwrites
`~/Desktop/robocupedu_camera_frame.jpg` with
the captured image. Press `q` to close the preview. Set `RECEPTIONIST_VLM_MODEL` to
use another local Ollama vision model. The model is prompted to return only
shirt color, glasses (yes/no), hair color, and clothing pattern (checkered,
plain, striped, or graphic T; `unclear` when not visible). It requests the four
fields together as JSON, prints them in a fixed labeled format, and asks the
language model to paraphrase only those attributes into one paragraph. Set
`RECEPTIONIST_LLM_MODEL` to use another locally installed language model.

The receptionist prototype starts directly after the “Hey Jarvis” wake word.
It first navigates to door pose `(-1.2464, -5.5788, -1.5734 rad)`. After name
confirmation it stores the confirmed name with the next `guest_N` identifier,
says “Please follow me,” then navigates to table pose
`(-1.22878, -4.03087, 0.12896 rad)`. Nav2, robot bringup, odometry, TF, and
AMCL localization must be active; the robot must be localized at its true
current position. Person detection, pose estimation, chair occupancy, and guest identity require
the RealSense camera and vision node. The seating-area waypoint is fixed; the
camera selects a free chair or couch visible from that location. Update the pose constants in `task_manager/main.py` if
the map waypoints change.
