# Kachaka × Autoware Core MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive Kachaka autonomously using the Autoware Core stack (NDT localization / lanelet2 planning / simple_pure_pursuit control / AD-API). The MVP target is: pick a single 2D Goal Pose in the standard Autoware RViz UI and have Kachaka reach it.

**Architecture:** Add four new packages under `ros2/` in the kachaka-api repository (`kachaka_autoware_bridge`, `kachaka_autoware_vehicle_interface`, `kachaka_autoware_description`, `kachaka_autoware_maps`) and connect the existing `kachaka_grpc_ros2_bridge` to Autoware Core through a Vehicle Interface node. The Vehicle Interface is responsible for converting `autoware_control_msgs/Control` (Ackermann) to `geometry_msgs/Twist` (differential drive), publishing `/vehicle/status/velocity_status`, running a simple Operation Mode state machine, and auto-enabling ManualControl.

**Tech Stack:** ROS 2 Jazzy / C++17 / `rclcpp` / `ament_cmake_auto` / gtest (`ament_cmake_gtest`) / launch_xml / Autoware Core (autoware_core_localization / planning / control / api) / autoware_rviz_plugins / ouster-ros driver / Vector Map Builder (external) / lio_sam or fast_lio or glim (external, picked in M0).

**User-side hardware / environment prerequisites:**
- Kachaka unit: 192.168.1.91, gRPC API port 26400, SW 3.16+
- Jetson Thor (Ubuntu 24.04 + ROS 2 Jazzy, ROS_DOMAIN_ID=123, ws=`~/ros/jazzy`)
- Ouster OS-1 128 (already mounted on the shelf, or to be mounted)
- Development PC (Jazzy + RViz2, same ROS_DOMAIN_ID)
- **The Kachaka built-in 2D LiDAR is not used** (this stack does not depend on Kachaka's built-in SLAM, mapping, or localization)

**Repository prerequisites:**
- `~/src/kachaka-api` (this repository)
- `~/src/autoware_core` (already cloned)
- Other dependency packages cloned under `~/ros/jazzy/src/`

---

## File Structure

### New packages (inside the kachaka-api repository)

```
ros2/
├── kachaka_autoware_bridge/                              # meta + integrated launch
│   ├── package.xml                                       # depends on the other 3 packages + autoware_core_*
│   ├── CMakeLists.txt
│   └── launch/
│       └── kachaka_autoware.launch.xml                   # all-in-one entry: bridge + full Autoware Core
│
├── kachaka_autoware_vehicle_interface/                   # core component
│   ├── package.xml                                       # depends on rclcpp / autoware_control_msgs / autoware_vehicle_msgs / autoware_adapi_v1_msgs / geometry_msgs / nav_msgs / std_srvs
│   ├── CMakeLists.txt
│   ├── include/kachaka_autoware_vehicle_interface/
│   │   ├── control_to_twist_converter.hpp               # Control msg -> Twist conversion (pure function)
│   │   ├── operation_mode_state_machine.hpp             # simple STOP / AUTONOMOUS state machine
│   │   ├── velocity_status_publisher.hpp                # Odometry -> VelocityReport conversion
│   │   └── vehicle_interface_node.hpp                   # rclcpp::Node subclass
│   ├── src/
│   │   ├── control_to_twist_converter.cpp
│   │   ├── operation_mode_state_machine.cpp
│   │   ├── velocity_status_publisher.cpp
│   │   ├── vehicle_interface_node.cpp                   # composes submodules and wires ROS I/O
│   │   └── main.cpp                                     # rclcpp::spin
│   ├── launch/
│   │   └── vehicle_interface.launch.xml
│   ├── config/
│   │   └── vehicle_interface.param.yaml
│   └── test/
│       ├── CMakeLists.txt                                # standalone test CMake fragment
│       ├── test_control_to_twist_converter.cpp          # boundary-value tests for pure logic
│       ├── test_operation_mode_state_machine.cpp        # state transition tests
│       └── test_velocity_status_publisher.cpp           # Odometry -> VelocityReport conversion tests
│
├── kachaka_autoware_description/                         # URDF + vehicle_info
│   ├── package.xml                                       # depends on kachaka_description / xacro / ouster_description
│   ├── CMakeLists.txt
│   ├── urdf/
│   │   ├── kachaka_autoware.urdf.xacro                  # includes kachaka.urdf.xacro and adds the shelf
│   │   └── shelf_with_ouster.urdf.xacro                 # static joint chain base_link -> shelf_dock_link -> os1_sensor
│   ├── config/
│   │   └── vehicle_info.param.yaml                      # virtual values tuned for differential drive
│   └── launch/
│       └── robot_description.launch.py                  # launches robot_state_publisher
│
└── kachaka_autoware_maps/                                # samples + instructions
    ├── package.xml
    ├── CMakeLists.txt
    └── README.md                                         # M0 instructions (pointing at user_dir/maps)
```

### Responsibilities (split by responsibility, not technical layer)

- **`control_to_twist_converter`**: pure function `Twist convert(const Control&, params)`. No ROS dependency. Optimised for testability.
- **`operation_mode_state_machine`**: pure class `OperationModeStateMachine`. State transition logic; `get_state()`, `request_autonomous()`, `request_stop()`. Independent of `rclcpp::Node`.
- **`velocity_status_publisher`**: pure function `VelocityReport convert(const Odometry&)`. Uses ROS msgs but is node-independent.
- **`vehicle_interface_node`**: `rclcpp::Node` that owns the submodules above. Only wires subscribers/publishers/services/timers; contains no logic.

This separation lets each submodule be unit-tested with gtest. When `operation_mode_state_machine` is replaced by Universe's `autoware_command_mode_decider`, only the composition inside `vehicle_interface_node` needs to change.

### External packages to clone (procedure already in spec §10.2)

- Clone `autoware_rviz_plugins` into `~/ros/jazzy/src/autoware_rviz_plugins/`
- Clone `ouster-ros` into `~/ros/jazzy/src/ouster-ros/` (upstream: https://github.com/ouster-lidar/ouster-ros)
- Symlink (or directly clone) `~/src/autoware_core` to `~/ros/jazzy/src/autoware_core`

---

## Milestone-to-Task Mapping

| Milestone | Tasks |
|---|---|
| M-1 Environment setup | Task 1 |
| M0 Pre-work | Task 2-5 |
| M1 Sensor integration | Task 6-13 |
| M2 Localization | Task 14-19 |
| M3 Vehicle Interface | Task 20-37 |
| M4 Planning | Task 38-42 |
| M5 Closed loop + verification | Task 43-46 |

---

## Tasks

### Task 1: Workspace bootstrap (M-1)

**Purpose:** Populate `~/ros/jazzy` on Thor with the required repositories, install apt dependencies, and reach a state where autoware_core / kachaka-api / autoware_rviz_plugins / ouster-ros can all be built. This is environment setup, not code writing.

**Files:**
- No changes (only external repo clones and apt installs)

- [ ] **Step 1: Install required apt dependencies (same list as jazzy_build_caveats.md)**

Run:
```bash
sudo apt update
sudo apt install -y \
  protobuf-compiler-grpc \
  libgrpc++-dev \
  nlohmann-json3-dev \
  ros-jazzy-xacro \
  ros-jazzy-rviz2 \
  ros-jazzy-tf2-ros \
  ros-jazzy-pcl-ros \
  ros-jazzy-pcl-conversions \
  ros-jazzy-perception-pcl \
  python3-colcon-common-extensions
```

Expected result: every package finishes with `Setting up ...` and no errors.

- [ ] **Step 2: Create `~/ros/jazzy/src` and link autoware_core / kachaka-api into it**

Run:
```bash
mkdir -p ~/ros/jazzy/src
cd ~/ros/jazzy/src
ln -sf ~/src/autoware_core autoware_core
ln -sf ~/src/kachaka-api kachaka-api
```

Expected result: `ls -la ~/ros/jazzy/src/` shows the two symlinks.

- [ ] **Step 3: Clone `autoware_rviz_plugins`**

Run:
```bash
cd ~/ros/jazzy/src
git clone https://github.com/autowarefoundation/autoware_rviz_plugins.git
```

Expected result: `ls ~/ros/jazzy/src/autoware_rviz_plugins/package.xml` exists.

- [ ] **Step 4: Clone `ouster-ros` (ros2 branch)**

Run:
```bash
cd ~/ros/jazzy/src
git clone -b ros2 https://github.com/ouster-lidar/ouster-ros.git
git -C ouster-ros submodule update --init --recursive
```

Expected result: `ls ~/ros/jazzy/src/ouster-ros/ouster_ros/package.xml` exists.

- [ ] **Step 5: Resolve package dependencies with rosdep**

Run:
```bash
cd ~/ros/jazzy
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -y --rosdistro=jazzy
```

Expected result: `All required rosdeps installed successfully`

- [ ] **Step 6: Generate gen-src for kachaka_grpc_ros2_bridge**

Run:
```bash
cd ~/ros/jazzy/src/kachaka-api
mkdir -p ros2/kachaka_grpc_ros2_bridge/gen-src
protoc -I protos \
  --grpc_out=ros2/kachaka_grpc_ros2_bridge/gen-src \
  --plugin=protoc-gen-grpc=/usr/bin/grpc_cpp_plugin \
  --cpp_out=ros2/kachaka_grpc_ros2_bridge/gen-src \
  protos/kachaka-api.proto
ls ros2/kachaka_grpc_ros2_bridge/gen-src/
```

Expected result: 4 files: `kachaka-api.grpc.pb.cc kachaka-api.grpc.pb.h kachaka-api.pb.cc kachaka-api.pb.h`.

- [ ] **Step 7: Baseline build (kachaka_interfaces / kachaka_description / kachaka_grpc_ros2_bridge / minimum autoware_core set)**

Run:
```bash
cd ~/ros/jazzy
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to \
  kachaka_grpc_ros2_bridge \
  autoware_core \
  autoware_rviz_plugins
```

Expected result: `Summary: ... packages finished` with zero errors. Warnings are tolerated.

- [ ] **Step 8: Smoke test — kachaka_grpc_ros2_bridge starts up**

Run:
```bash
cd ~/ros/jazzy
source install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka &
sleep 5
ros2 topic list | grep -E "(kachaka|tf)"
kill %1 2>/dev/null
```

Expected result: topics such as `/kachaka/odometry/odometry`, `/kachaka/imu/imu`, `/tf` appear.

- [ ] **Step 9: Commit (only if there are changes)**

Skip if there are no changes. This task is external-dependency setup and normally produces no repository changes.

---

### Task 2: M0-A — Build the pointcloud_map

**Purpose:** Use the OS-1 128 alone to map the home and produce `pointcloud_map.pcd` + `pointcloud_map/metadata.yaml`. The Kachaka 2D LiDAR cannot be used in this pipeline.

**Files:**
- Output: `~/maps/kachaka_home/pointcloud_map.pcd`
- Output: `~/maps/kachaka_home/pointcloud_map/metadata.yaml`

- [ ] **Step 1: Pick one SLAM tool and clone it into `~/ros/jazzy/src`**

Candidates (spec §3.3 / 16):
- `glim` (Jazzy-compatible, 3D LiDAR + IMU, recommended): https://github.com/koide3/glim
- `fast_lio`: https://github.com/hku-mars/FAST_LIO
- `lio_sam`: https://github.com/TixiaoShan/LIO-SAM

If using `glim`:
```bash
cd ~/ros/jazzy/src
git clone https://github.com/koide3/glim.git
git clone https://github.com/koide3/glim_ros2.git
```

Expected result: clone succeeds. Read the README and install its apt dependencies.

- [ ] **Step 2: Build the SLAM tool**

Run:
```bash
cd ~/ros/jazzy
colcon build --symlink-install --packages-up-to glim_ros2
```

Expected result: build succeeds.

- [ ] **Step 3: Connect OS-1 128 to Thor over Ethernet and publish points via the ouster-ros driver**

Run:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch ouster_ros sensor.launch.xml \
  sensor_hostname:=os-XXXXXX.local \
  metadata:=/tmp/os1_meta.json &
sleep 5
ros2 topic hz /ouster/points
```

Expected result: points publish at roughly 10-20 Hz. `/ouster/imu` also publishes.

The OS-1 hostname / IP is user-specific; replace `os-XXXXXX.local` with the value for your unit.

- [ ] **Step 4: Mount OS-1 on Kachaka and drive a mapping pass manually (push or teleop)**

Procedure (manual, no code):
1. Set Kachaka to `set_manual_control_enabled(true)`, then push it by hand or teleop via `kachaka_grpc_ros2_bridge/manual_control`.
2. Cover every navigable area of the home (straight lines, turns, room corners).
3. Run the SLAM tool to build the pointcloud map during the drive (with glim, run `glim_rosnode` and `rosbag record` in parallel).
4. Close the loop by returning to the starting point.

Expected result: the rosbag contains a complete recording of OS-1 points and IMU.

- [ ] **Step 5: Save pointcloud_map.pcd and write metadata.yaml**

For glim:
```bash
mkdir -p ~/maps/kachaka_home/pointcloud_map
# Save glim's output to ~/maps/kachaka_home/pointcloud_map.pcd
# Write metadata.yaml in the format autoware_map_loader expects
cat > ~/maps/kachaka_home/pointcloud_map/metadata.yaml <<'EOF'
x_resolution: 50.0
y_resolution: 50.0
A.pcd: [0, 0]
EOF
```

Expected result: `~/maps/kachaka_home/` contains `pointcloud_map.pcd` and `pointcloud_map/metadata.yaml`. The pcd is roughly 100 MB to a few GB depending on indoor coverage.

Implementation note: Autoware's `autoware_map_loader` expects split tile maps, so the metadata.yaml above is the minimal config that treats the **single map as one tile**. See the `autoware_map_loader` docs for details.

- [ ] **Step 6: No commit — maps live outside the repository**

Reason: pointcloud_map.pcd is too large for the repository. The location is documented in `kachaka_autoware_maps/README.md` (created in Task 5).

---

### Task 3: M0-B — Build the lanelet2 vector_map

**Purpose:** Use Vector Map Builder to draw minimal lanes over the navigable area of the home and produce `lanelet2_map.osm` + `map_projector_info.yaml`. Use the same local projection origin as the pointcloud_map.

**Files:**
- Output: `~/maps/kachaka_home/lanelet2_map.osm`
- Output: `~/maps/kachaka_home/map_projector_info.yaml`

- [ ] **Step 1: Open Vector Map Builder (TIER IV web tool)**

Open https://tools.tier4.jp/vector_map_builder_ll2/ in a browser (or look up the latest URL in the TIER IV docs).

- [ ] **Step 2: Import pointcloud_map.pcd as the background**

Use Vector Map Builder's "Load PCD" to load the `pointcloud_map.pcd` produced in Task 2.

Expected result: the home pointcloud is displayed on screen.

- [ ] **Step 3: Draw minimal lanes through the navigable area**

Steps:
1. Draw 1-2 straight lanes from one room to another (keep it minimal).
2. Each lane width = Kachaka body width (0.387 m) + margin, around `0.6 m`.
3. Speed limit: `0.3 m/s` (Kachaka's maximum linear velocity).

Expected result: a routable lanelet2 layout.

- [ ] **Step 4: Export lanelet2_map.osm**

Use Vector Map Builder's "Export" to download `lanelet2_map.osm` and save it as `~/maps/kachaka_home/lanelet2_map.osm`.

Expected result: file saved (typically a few KB).

- [ ] **Step 5: Create map_projector_info.yaml**

Run:
```bash
cat > ~/maps/kachaka_home/map_projector_info.yaml <<'EOF'
projector_type: Local
vertical_datum: WGS84
EOF
```

Note: `projector_type: Local` is the GNSS-free indoor setup, assuming "Local Cartesian" was selected during Vector Map Builder export.

Expected result: file saved.

- [ ] **Step 6: Verify origin alignment with pointcloud_map**

Manual verification:
1. In RViz, overlay pointcloud_map.pcd (via pcl_ros `pcd_to_pointcloud`) and lanelet2_map (via `autoware_lanelet2_map_visualizer`) in the same `map` frame.
2. Visually confirm that wall positions and lane positions are aligned.
3. If misaligned, re-adjust in Vector Map Builder and re-export.

Expected result: the pointcloud_map walls and lanelet2 lanes are aligned.

- [ ] **Step 7: No commit**

Maps are not included in the repository.

---

### Task 4: M0-C — OS-1 physical mounting and calibration measurements

**Purpose:** Physically mount the Ouster OS-1 on top of the shelf and measure the `shelf_top -> os1_sensor` offset and the rough shelf dimensions. The values are transferred into the params of Task 6's `_shelf_3tier.urdf.xacro` and the `<origin>` of Task 7's `_ouster_os1.urdf.xacro`.

**Files:**
- Output: a notes file (numbers to copy into the Task 6 shelf dimension defaults and the Task 7 OS-1 mount origin)

- [ ] **Step 1: Mount OS-1 near the centre of the shelf top**

Physical task (no code). Mounting method is up to the user. Constraints:
- The "forward" marker of the sensor must align with Kachaka's forward direction.
- Keep it level; tilt degrades NDT matching accuracy.
- Minimise xy offset from the centre of the shelf top to reduce model error.

- [ ] **Step 2: Measure with a tape measure**

Manual measurements (numbers to feed into the URDF):
- **Shelf dimensions**:
  - depth (x, fore-aft): measure with a tape; check against default 0.32 m
  - width (y, left-right): check against default 0.38 m
  - height (z, from shelf bottom to top board upper surface): check against default 0.50 m
- **OS-1 mounting offset relative to the shelf top** (`shelf_top` frame):
  - x: forward distance from the shelf-top centre to the OS-1 mount centre (0 if centred)
  - y: lateral distance (same convention)
  - z: height from the shelf top board upper surface to the OS-1 sensor origin (cylinder bottom). Normally 0 (mounted directly).
  - roll/pitch/yaw: normally 0
- **base_footprint -> docking_link -> shelf_base** are all fixed at (0,0,0) (the kachaka_description / _shelf_3tier convention). The URDF computes the OS-1 lidar height from base_footprint as `docking_link height + shelf height + OS-1 body_height + lidar_to_sensor_z`.

Expected result: numerical notes (e.g. `shelf depth=0.32 / width=0.38 / height=0.50, os1_offset_xyz=0,0,0, rpy=0,0,0`).

- [ ] **Step 3: Confirm measurement accuracy**

Initially ±2 cm / ±2 deg is enough. Re-tune in M2 if NDT does not converge.

Expected result: notes finalised.

- [ ] **Step 4: No commit**

Physical work only. The values are transferred into the Task 6 shelf dimension defaults and the Task 7 OS-1 origin.

---

### Task 5: Create the kachaka_autoware_maps package and map preparation guide

**Purpose:** Map files live outside the repository (`~/maps/kachaka_home/`), so the `kachaka_autoware_maps` package only ships the preparation guide.

**Files:**
- Create: `ros2/kachaka_autoware_maps/package.xml`
- Create: `ros2/kachaka_autoware_maps/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_maps/README.md`

- [ ] **Step 1: Create package.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_maps/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format2.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>kachaka_autoware_maps</name>
  <version>0.0.0</version>
  <description>Documentation and helper scripts for Kachaka Autoware map preparation (M0). Map files themselves live outside the repository.</description>
  <maintainer email="support@kachaka.life">Kachaka Customer Support</maintainer>
  <license>Apache License 2.0</license>
  <author email="support@kachaka.life">Kachaka Customer Support</author>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 2: Create CMakeLists.txt**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_maps/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.5)
project(kachaka_autoware_maps)

find_package(ament_cmake REQUIRED)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
```

- [ ] **Step 3: Write README.md**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_maps/README.md`:

```markdown
# kachaka_autoware_maps

Guide for preparing the maps (pointcloud_map.pcd and lanelet2_map.osm) needed to drive Kachaka under Autoware Core. The map files themselves are not committed to the repository and live under the user's `~/maps/<location_name>/`.

## Prerequisites

- OS-1 128 (or an equivalent 3D LiDAR) is mounted on Kachaka.
- The ouster-ros driver can publish points.
- **The Kachaka 2D LiDAR is not used** (this stack does not depend on it).

## Directory layout

```
~/maps/<location_name>/
├── pointcloud_map.pcd
├── pointcloud_map/
│   └── metadata.yaml
├── lanelet2_map.osm
└── map_projector_info.yaml
```

Pass these paths to the `lanelet2_map_path`, `pointcloud_map_path`, `pointcloud_map_metadata_path`, and `map_projector_info_path` arguments of `autoware_core_map.launch.xml`.

## 1. Building pointcloud_map.pcd

Run 3D SLAM with the OS-1 alone (recommended tool: `glim`). See `docs/superpowers/specs/2026-05-02-kachaka-autoware-core-design.md` §3.3 for details.

## 2. Building lanelet2_map.osm

Use [TIER IV Vector Map Builder](https://tools.tier4.jp/vector_map_builder_ll2/) to draw lanes on top of pointcloud_map.pcd. **Export with the same local projection origin as the pointcloud_map.**

## 3. map_projector_info.yaml

```yaml
projector_type: Local
vertical_datum: WGS84
```

## 4. pointcloud_map/metadata.yaml (single-tile usage)

```yaml
x_resolution: 50.0
y_resolution: 50.0
A.pcd: [0, 0]
```

For implementation details, see the `autoware_map_loader` documentation.
```

- [ ] **Step 4: Confirm the build passes**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_maps
```

Expected result: `Summary: 1 package finished`

- [ ] **Step 5: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_maps/
git commit -m "$(cat <<'EOF'
feat(maps): add kachaka_autoware_maps package with M0 map preparation guide

Map files (pointcloud_map.pcd, lanelet2_map.osm) live outside the
repository in ~/maps/<location_name>/. This package only provides the
README documenting how to build them with OS-1 + Vector Map Builder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected result: commit succeeds.

---

### Task 6: Enhance kachaka_description — add the genuine 3-tier shelf macro

**Purpose:** The Kachaka 3-tier shelf is a first-party accessory, so it is added to the `kachaka_description` package as `_shelf_3tier.urdf.xacro`. The existing `_kachaka.urdf.xacro` / `_values.urdf.xacro` / `kachaka.urdf.xacro` must not change in a breaking way (existing users' URDF output stays the same). The shelf is a macro attached to `docking_link` so it follows docking and lift motion.

**Files:**
- Modify: `ros2/kachaka_description/urdf/_materials.urdf.xacro` (add shelf materials)
- Create: `ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro`

- [ ] **Step 1: Add materials**

Append the following two materials to `/home/youtalk/src/kachaka-api/ros2/kachaka_description/urdf/_materials.urdf.xacro` immediately before `</robot>`:

```xml
  <material name="shelf_board">
    <color rgba="0.85 0.78 0.65 1.0" />
  </material>
  <material name="shelf_post">
    <color rgba="0.15 0.15 0.15 1.0" />
  </material>
```

Final form:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro">
  <material name="body">
    <color rgba="0.2175 0.2355 0.246 1.0" />
  </material>
  <material name="tire">
    <color rgba="0.3 0.3 0.3 1.0" />
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0" />
  </material>
  <material name="shelf_board">
    <color rgba="0.85 0.78 0.65 1.0" />
  </material>
  <material name="shelf_post">
    <color rgba="0.15 0.15 0.15 1.0" />
  </material>
</robot>
```

- [ ] **Step 2: Create `_shelf_3tier.urdf.xacro`**

`/home/youtalk/src/kachaka-api/ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="shelf_3tier" xmlns:xacro="http://ros.org/wiki/xacro">
  <!--
    Macro for the genuine Kachaka 3-tier shelf.
    Dimensions are approximate; pass measured values via the macro params.
    `parent` is normally docking_link so the shelf follows the docking lift.
  -->
  <xacro:macro name="shelf_3tier"
               params="parent
                       *origin
                       shelf_name:=shelf
                       depth:=0.32
                       width:=0.38
                       height:=0.50
                       board_thickness:=0.015
                       post_size:=0.020">

    <!-- bottom-face center of the shelf is at the origin block passed by the caller -->
    <link name="${shelf_name}_base_link"/>
    <joint name="${shelf_name}_base_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${shelf_name}_base_link"/>
      <xacro:insert_block name="origin"/>
    </joint>

    <!-- 3 horizontal boards: bottom (z=0), middle, top -->
    <xacro:macro name="_shelf_board" params="board_name z">
      <link name="${shelf_name}_${board_name}_board">
        <visual>
          <origin xyz="0 0 ${z + board_thickness/2}" rpy="0 0 0"/>
          <geometry>
            <box size="${depth} ${width} ${board_thickness}"/>
          </geometry>
          <material name="shelf_board"/>
        </visual>
        <collision>
          <origin xyz="0 0 ${z + board_thickness/2}" rpy="0 0 0"/>
          <geometry>
            <box size="${depth} ${width} ${board_thickness}"/>
          </geometry>
        </collision>
      </link>
      <joint name="${shelf_name}_${board_name}_board_joint" type="fixed">
        <parent link="${shelf_name}_base_link"/>
        <child link="${shelf_name}_${board_name}_board"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
      </joint>
    </xacro:macro>

    <xacro:_shelf_board board_name="bottom" z="0"/>
    <xacro:_shelf_board board_name="middle" z="${(height - board_thickness) / 2.0}"/>
    <xacro:_shelf_board board_name="top"    z="${height - board_thickness}"/>

    <!-- 4 corner posts (vertical) -->
    <xacro:macro name="_shelf_post" params="post_name x y">
      <link name="${shelf_name}_${post_name}_post">
        <visual>
          <origin xyz="${x} ${y} ${height/2.0}" rpy="0 0 0"/>
          <geometry>
            <box size="${post_size} ${post_size} ${height}"/>
          </geometry>
          <material name="shelf_post"/>
        </visual>
        <collision>
          <origin xyz="${x} ${y} ${height/2.0}" rpy="0 0 0"/>
          <geometry>
            <box size="${post_size} ${post_size} ${height}"/>
          </geometry>
        </collision>
      </link>
      <joint name="${shelf_name}_${post_name}_post_joint" type="fixed">
        <parent link="${shelf_name}_base_link"/>
        <child link="${shelf_name}_${post_name}_post"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
      </joint>
    </xacro:macro>

    <xacro:_shelf_post post_name="fl" x="${ (depth - post_size)/2.0}" y="${ (width - post_size)/2.0}"/>
    <xacro:_shelf_post post_name="fr" x="${ (depth - post_size)/2.0}" y="${-(width - post_size)/2.0}"/>
    <xacro:_shelf_post post_name="bl" x="${-(depth - post_size)/2.0}" y="${ (width - post_size)/2.0}"/>
    <xacro:_shelf_post post_name="br" x="${-(depth - post_size)/2.0}" y="${-(width - post_size)/2.0}"/>

    <!-- shelf_top: a fixed anchor link on the top board's upper surface for payloads -->
    <link name="${shelf_name}_top"/>
    <joint name="${shelf_name}_top_joint" type="fixed">
      <parent link="${shelf_name}_base_link"/>
      <child link="${shelf_name}_top"/>
      <origin xyz="0 0 ${height}" rpy="0 0 0"/>
    </joint>
  </xacro:macro>
</robot>
```

- [ ] **Step 3: Confirm the existing `kachaka.urdf.xacro` output is unchanged**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_description
source install/setup.bash
xacro src/kachaka-api/ros2/kachaka_description/robot/kachaka.urdf.xacro > /tmp/kachaka_after.urdf
grep -c "<link" /tmp/kachaka_after.urdf
grep -c "<joint" /tmp/kachaka_after.urdf
grep "shelf" /tmp/kachaka_after.urdf || echo "no shelf in default kachaka — OK"
```

Expected result: existing link/joint counts are unchanged. `shelf` does not appear in the output (kachaka.urdf.xacro does not include the shelf macro).

- [ ] **Step 4: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_description/urdf/_materials.urdf.xacro \
        ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro
git commit -m "$(cat <<'EOF'
feat(description): add 3-tier Kachaka shelf macro

_shelf_3tier.urdf.xacro provides a parameterized 3-tier shelf macro.
Default dimensions are approximate; pass explicit params to override.
Existing kachaka.urdf.xacro output is unchanged (the macro is opt-in).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: kachaka_autoware_description package + Ouster + integrated URDF

**Purpose:** Create a new package with the Ouster OS-1 macro and a complete URDF that integrates Kachaka + 3-tier shelf + OS-1.

**Files:**
- Create: `ros2/kachaka_autoware_description/package.xml`
- Create: `ros2/kachaka_autoware_description/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_description/urdf/_ouster_os1.urdf.xacro`
- Create: `ros2/kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro`

- [ ] **Step 1: package.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format2.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>kachaka_autoware_description</name>
  <version>0.0.0</version>
  <description>Composite URDF for Kachaka with the 3-tier shelf and a roof-mounted Ouster OS-1 128, plus Autoware vehicle_info parameters tuned for differential drive.</description>
  <maintainer email="support@kachaka.life">Kachaka Customer Support</maintainer>
  <license>Apache License 2.0</license>
  <author email="support@kachaka.life">Kachaka Customer Support</author>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <build_depend>kachaka_description</build_depend>
  <build_depend>xacro</build_depend>
  <exec_depend>kachaka_description</exec_depend>
  <exec_depend>xacro</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>launch_xml</exec_depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 2: CMakeLists.txt**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.5)
project(kachaka_autoware_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
```

- [ ] **Step 3: Create `_ouster_os1.urdf.xacro`**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/urdf/_ouster_os1.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="ouster_os1" xmlns:xacro="http://ros.org/wiki/xacro">
  <!--
    Simplified Ouster OS-1 (128) model. Replace with the official ouster_description if available.
    body: cylinder (85 mm diameter, 73.5 mm tall).
    os1_lidar / os1_imu frame offsets are taken from the Ouster ICD.
  -->
  <xacro:macro name="ouster_os1"
               params="parent
                       name:=os1
                       *origin
                       body_radius:=0.0425
                       body_height:=0.0735
                       lidar_to_sensor_z:=0.03618
                       imu_offset_x:=0.006253
                       imu_offset_y:=-0.011775
                       imu_offset_z:=0.007645">

    <link name="${name}_sensor">
      <visual>
        <origin xyz="0 0 ${body_height/2.0}" rpy="0 0 0"/>
        <geometry>
          <cylinder length="${body_height}" radius="${body_radius}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <origin xyz="0 0 ${body_height/2.0}" rpy="0 0 0"/>
        <geometry>
          <cylinder length="${body_height}" radius="${body_radius}"/>
        </geometry>
      </collision>
    </link>
    <joint name="${name}_mount_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${name}_sensor"/>
      <xacro:insert_block name="origin"/>
    </joint>

    <link name="${name}_lidar"/>
    <joint name="${name}_lidar_joint" type="fixed">
      <parent link="${name}_sensor"/>
      <child link="${name}_lidar"/>
      <origin xyz="0 0 ${lidar_to_sensor_z}" rpy="0 0 0"/>
    </joint>

    <link name="${name}_imu"/>
    <joint name="${name}_imu_joint" type="fixed">
      <parent link="${name}_sensor"/>
      <child link="${name}_imu"/>
      <origin xyz="${imu_offset_x} ${imu_offset_y} ${imu_offset_z}" rpy="0 0 0"/>
    </joint>
  </xacro:macro>
</robot>
```

Note: `_ouster_os1.urdf.xacro` references the `black` material from `_materials.urdf.xacro`, so the include order must be kachaka first, then ouster (handled in Step 4).

- [ ] **Step 4: Create `kachaka_with_shelf.urdf.xacro`**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="kachaka_with_shelf" xmlns:xacro="http://ros.org/wiki/xacro">

  <!-- materials/values come from kachaka_description so kachaka_robot and shelf macros work -->
  <xacro:include filename="$(find kachaka_description)/urdf/_materials.urdf.xacro"/>
  <xacro:include filename="$(find kachaka_description)/urdf/_values.urdf.xacro"/>
  <xacro:include filename="$(find kachaka_description)/urdf/_kachaka.urdf.xacro"/>
  <xacro:include filename="$(find kachaka_description)/urdf/_shelf_3tier.urdf.xacro"/>
  <xacro:include filename="$(find kachaka_autoware_description)/urdf/_ouster_os1.urdf.xacro"/>

  <!-- root -->
  <link name="base_footprint"/>

  <!-- Kachaka chassis (creates base_link, wheels, docking_link, etc.) -->
  <xacro:kachaka_robot parent="base_footprint">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:kachaka_robot>

  <!--
    3-tier shelf attached to docking_link with z=0.115 m so the shelf
    bottom sits on top of the solenoid (cylinder center 0.1075 + half
    length 0.0075). docking_link is a prismatic child of base_link, so
    the shelf still rises and falls with the lift (up to +0.012 m).
  -->
  <xacro:shelf_3tier parent="docking_link">
    <origin xyz="0 0 0.115" rpy="0 0 0"/>
  </xacro:shelf_3tier>

  <!-- Ouster OS-1 mounted at the shelf top -->
  <xacro:ouster_os1 parent="shelf_top">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:ouster_os1>

</robot>
```

- [ ] **Step 5: Build and xacro-expansion check**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_description
source install/setup.bash
xacro src/kachaka-api/ros2/kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro > /tmp/kachaka_with_shelf.urdf
echo "xacro exit: $?"
grep -c "<link " /tmp/kachaka_with_shelf.urdf
grep -c "<joint " /tmp/kachaka_with_shelf.urdf
grep -E "shelf_(base_link|top|fl_post|bottom_board)" /tmp/kachaka_with_shelf.urdf
grep -E "os1_(sensor|lidar|imu)" /tmp/kachaka_with_shelf.urdf
```

Expected result:
- `xacro exit: 0`
- link count >= 18 (Kachaka 9 existing links + shelf 9 links + os1 3 links)
- joint count >= 17
- At least one `shelf_*` and one `os1_*` link match grep.

- [ ] **Step 6: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_description/
git commit -m "$(cat <<'EOF'
feat(description): kachaka_autoware_description with shelf + OS-1 URDF

- _ouster_os1.urdf.xacro: simplified OS-1 128 model (cylinder body +
  lidar/imu frames per Ouster ICD offsets)
- kachaka_with_shelf.urdf.xacro: full robot URDF combining Kachaka
  chassis, 3-tier shelf attached to docking_link, and OS-1 on shelf_top

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: vehicle_info.param.yaml and robot_description.launch.py

**Purpose:** Create the `vehicle_info` consumed by `simple_pure_pursuit` and planning, tuned for the differential-drive Kachaka. Also provide a launch file that starts `robot_state_publisher`.

**Files:**
- Create: `ros2/kachaka_autoware_description/config/vehicle_info.param.yaml`
- Create: `ros2/kachaka_autoware_description/launch/robot_description.launch.py`

- [ ] **Step 1: Create vehicle_info.param.yaml (transcribe values from spec §9.3)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/config/vehicle_info.param.yaml`:

```yaml
/**:
  ros__parameters:
    wheel_radius: 0.045
    wheel_width: 0.025
    wheel_base: 0.30
    wheel_tread: 0.20
    # Footprint overhangs from base_link origin (body collision center
    # 0.0435,0,0.0475, size 0.387 x 0.240 x 0.095)
    front_overhang: 0.237
    rear_overhang: 0.150
    left_overhang: 0.120
    right_overhang: 0.120
    vehicle_height: 1.20
    max_steer_angle: 1.5708
```

- [ ] **Step 2: Create robot_description.launch.py**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/launch/robot_description.launch.py`:

```python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("kachaka_autoware_description")
    xacro_path = os.path.join(pkg, "urdf", "kachaka_with_shelf.urdf.xacro")

    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="", description="Robot namespace prefix"
    )

    robot_description = {
        "robot_description": Command(["xacro ", xacro_path]),
    }

    return LaunchDescription(
        [
            namespace_arg,
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                namespace=LaunchConfiguration("namespace"),
                parameters=[robot_description],
                output="screen",
            ),
        ]
    )
```

- [ ] **Step 3: Confirm the build**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_description
```

Expected result: build succeeds.

- [ ] **Step 4: Verify the launch file runs standalone**

Run:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_description robot_description.launch.py &
sleep 3
ros2 topic echo --once /robot_description | head -3
ros2 run tf2_tools view_frames -o /tmp/frames &
sleep 5
kill %1 %2 2>/dev/null
```

Expected result: the URDF flows on `/robot_description`. The TF tree shows base_footprint -> base_link -> docking_link -> shelf_base_link -> shelf_top -> os1_sensor. `ros2 run tf2_ros tf2_echo base_footprint os1_sensor` returns a transform (z = 0 from base_link -> docking_link + 0.115 to the solenoid top + 0.50 shelf height = 0.615 m; `base_footprint -> os1_lidar` adds 0.03618 = ~0.651 m). Passing `use_joint_state_publisher:=true` also fills in docking_link / wheel TFs.

- [ ] **Step 5: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_description/config/ ros2/kachaka_autoware_description/launch/
git commit -m "$(cat <<'EOF'
feat(description): add vehicle_info.param.yaml and robot_description launch

vehicle_info uses virtual wheel_base (0.30 m) for differential-drive
Kachaka; tunable in M5. robot_description.launch.py runs robot_state
_publisher with the augmented Kachaka+OS-1 URDF.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: ouster-ros launch wrapper

**Purpose:** Provide a launch file that publishes OS-1 128 data on the topic name Autoware expects, `/sensing/lidar/top/pointcloud_raw_ex`.

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/sensor_ouster.launch.xml`
- Other files are created together with the `kachaka_autoware_bridge` package in Task 10.

This work is performed alongside the package creation in Task 10, so it is **merged into Task 10**.

---

### Task 10: kachaka_autoware_bridge package skeleton + sensor launch

**Purpose:** Create the meta package `kachaka_autoware_bridge` and add its first launch file (the OS-1 wrapper).

**Files:**
- Create: `ros2/kachaka_autoware_bridge/package.xml`
- Create: `ros2/kachaka_autoware_bridge/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_bridge/launch/sensor_ouster.launch.xml`

- [ ] **Step 1: package.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format2.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>kachaka_autoware_bridge</name>
  <version>0.0.0</version>
  <description>Top-level launch and integration glue connecting Kachaka gRPC ROS 2 bridge to Autoware Core.</description>
  <maintainer email="support@kachaka.life">Kachaka Customer Support</maintainer>
  <license>Apache License 2.0</license>
  <author email="support@kachaka.life">Kachaka Customer Support</author>
  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>launch_ros</exec_depend>
  <exec_depend>launch_xml</exec_depend>
  <exec_depend>kachaka_grpc_ros2_bridge</exec_depend>
  <exec_depend>kachaka_autoware_description</exec_depend>
  <exec_depend>kachaka_autoware_vehicle_interface</exec_depend>
  <exec_depend>kachaka_autoware_maps</exec_depend>

  <exec_depend>ouster_ros</exec_depend>
  <exec_depend>autoware_core_map</exec_depend>
  <exec_depend>autoware_core_localization</exec_depend>
  <exec_depend>autoware_core_planning</exec_depend>
  <exec_depend>autoware_core_control</exec_depend>
  <exec_depend>autoware_core_api</exec_depend>
  <exec_depend>autoware_core_vehicle</exec_depend>
  <exec_depend>autoware_default_adapi</exec_depend>
  <exec_depend>autoware_adapi_adaptors</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 2: CMakeLists.txt**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.5)
project(kachaka_autoware_bridge)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
```

- [ ] **Step 3: sensor_ouster.launch.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/launch/sensor_ouster.launch.xml`:

```xml
<?xml version="1.0"?>
<launch>
  <arg name="sensor_hostname" default="os-122000000000.local" description="Ouster OS-1 hostname or IP. Override per-robot."/>
  <arg name="metadata" default="/tmp/os1_metadata.json" description="Path to ouster metadata cache"/>

  <!-- Run ouster-ros driver, remap output to Autoware-expected topic name -->
  <include file="$(find-pkg-share ouster_ros)/launch/sensor.launch.xml">
    <arg name="sensor_hostname" value="$(var sensor_hostname)"/>
    <arg name="metadata" value="$(var metadata)"/>
  </include>

  <!-- Republish /ouster/points → /sensing/lidar/top/pointcloud_raw_ex -->
  <node pkg="topic_tools" exec="relay" name="ouster_to_autoware_relay">
    <param name="input_topic" value="/ouster/points"/>
    <param name="output_topic" value="/sensing/lidar/top/pointcloud_raw_ex"/>
    <param name="type" value="sensor_msgs/msg/PointCloud2"/>
    <param name="reliability" value="best_effort"/>
  </node>

  <!-- Republish /ouster/imu → /sensing/imu/imu (for future EKF imu input) -->
  <node pkg="topic_tools" exec="relay" name="ouster_imu_relay">
    <param name="input_topic" value="/ouster/imu"/>
    <param name="output_topic" value="/sensing/imu/imu"/>
    <param name="type" value="sensor_msgs/msg/Imu"/>
    <param name="reliability" value="best_effort"/>
  </node>
</launch>
```

Note: confirm the launch file name and arg names of `ouster-ros` from its upstream README. `sensor.launch.xml` is the name on the ros2 branch.

- [ ] **Step 4: Confirm the build**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

Expected result: build succeeds.

- [ ] **Step 5: Smoke-test the launch on its own (when an OS-1 is connected)**

Run:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge sensor_ouster.launch.xml \
  sensor_hostname:=<actual hostname> &
sleep 8
ros2 topic hz /sensing/lidar/top/pointcloud_raw_ex
kill %1
```

Expected result: points publish at roughly 10-20 Hz.

- [ ] **Step 6: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/
git commit -m "$(cat <<'EOF'
feat(bridge): scaffold kachaka_autoware_bridge with Ouster sensor launch

Wraps ouster-ros driver and remaps /ouster/points to the Autoware-
expected /sensing/lidar/top/pointcloud_raw_ex topic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: TF / pointcloud integration check on real hardware (M1 exit criteria)

**Purpose:** Run robot_state_publisher and the ouster driver, then confirm in RViz2 that the pointcloud appears in the `base_footprint` frame.

- [ ] **Step 1: Integrated startup (three manual terminals)**

Terminal 1 (Kachaka bridge):
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka
```

Terminal 2 (OS-1):
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge sensor_ouster.launch.xml sensor_hostname:=<actual hostname>
```

Terminal 3 (URDF):
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_description robot_description.launch.py
```

- [ ] **Step 2: Check the TF tree**

Run:
```bash
ros2 run tf2_ros tf2_echo base_footprint os1_sensor
ros2 run tf2_ros tf2_echo base_footprint shelf_top
ros2 run tf2_ros tf2_echo shelf_top os1_sensor
```

Expected result: `base_footprint -> os1_sensor` z = solenoid top 0.115 + shelf height 0.50 + os1 mount offset 0 = 0.615 m. `shelf_top -> os1_sensor` is the OS-1 mounting offset measured in Task 4 (all zero if mounted at the centre).

- [ ] **Step 3: Visualise the pointcloud in RViz2**

Run:
```bash
rviz2 &
```

RViz steps:
- Set Fixed Frame to `base_footprint`.
- Add -> PointCloud2 -> Topic `/sensing/lidar/top/pointcloud_raw_ex`.
- Add -> RobotModel -> Description Topic `/robot_description`.

Expected result: the Kachaka URDF and the OS-1 pointcloud (room pointcloud) are rendered in the same frame.

- [ ] **Step 4: Commit calibration tweaks (optional)**

Possible changes from this task are URDF calibration tweaks:
- Shelf dimension defaults: tune `depth` / `width` / `height` defaults in `kachaka_description/urdf/_shelf_3tier.urdf.xacro` to measured values.
- OS-1 mount origin: tune the `<origin>` of `<xacro:ouster_os1>` in `kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro` to the measured offset.

After tuning, commit:

```bash
cd ~/src/kachaka-api
git add ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro \
        ros2/kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro
git commit -m "fix(description): tune shelf and OS-1 calibration values

Adjusted from Task 11 visual inspection in RViz against the live OS-1
point cloud.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Skip if no changes are needed.

---

### Task 12: kachaka_autoware_vehicle_interface package skeleton

**Purpose:** Before writing logic via TDD, create an empty package that builds.

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/package.xml`
- Create: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/.gitkeep`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/.gitkeep`
- Create: `ros2/kachaka_autoware_vehicle_interface/test/.gitkeep`
- Create: `ros2/kachaka_autoware_vehicle_interface/launch/.gitkeep`
- Create: `ros2/kachaka_autoware_vehicle_interface/config/.gitkeep`

- [ ] **Step 1: package.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format2.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>kachaka_autoware_vehicle_interface</name>
  <version>0.0.0</version>
  <description>Vehicle Interface bridging Autoware Core control output to Kachaka manual_control/cmd_vel, with operation_mode state machine and velocity_status republishing.</description>
  <maintainer email="support@kachaka.life">Kachaka Customer Support</maintainer>
  <license>Apache License 2.0</license>
  <author email="support@kachaka.life">Kachaka Customer Support</author>

  <buildtool_depend>ament_cmake_auto</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>
  <depend>std_srvs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>autoware_control_msgs</depend>
  <depend>autoware_vehicle_msgs</depend>
  <depend>autoware_adapi_v1_msgs</depend>

  <test_depend>ament_cmake_gtest</test_depend>
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 2: CMakeLists.txt (minimal; no code yet)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(kachaka_autoware_vehicle_interface)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
```

- [ ] **Step 3: Empty directories + .gitkeep**

Run:
```bash
cd /home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface
mkdir -p include/kachaka_autoware_vehicle_interface src test launch config
touch include/kachaka_autoware_vehicle_interface/.gitkeep src/.gitkeep test/.gitkeep launch/.gitkeep config/.gitkeep
```

- [ ] **Step 4: Confirm the build (empty package still builds)**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

Expected result: `Summary: 1 package finished`

- [ ] **Step 5: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): scaffold kachaka_autoware_vehicle_interface

Empty package skeleton; node and tests are added in subsequent tasks
following TDD.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: M1 exit checkpoint (no-change commit)

**Purpose:** Confirm the M1 exit criteria: OS-1 publishes via ROS 2, the TF tree is complete, and the pointcloud is visible in RViz against the `base_footprint` frame.

- [ ] **Step 1: Run the checklist**

Manual checks:
1. `ros2 topic hz /sensing/lidar/top/pointcloud_raw_ex` reports 10-20 Hz.
2. `ros2 run tf2_ros tf2_echo base_footprint os1_sensor` returns a transform.
3. In RViz2 with fixed frame `base_footprint`, the pointcloud and Kachaka URDF align.

- [ ] **Step 2: No commit**

Verification only. Move on to M2.

---

### Task 14: autoware_core_localization launch wrapper

**Purpose:** Launch `autoware_core_localization` with the Kachaka inputs (OS-1 + Kachaka wheel_odometry).

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/localization.launch.xml`
- Create: `ros2/kachaka_autoware_bridge/config/pose_initializer.param.yaml`

- [ ] **Step 1: Create pose_initializer.param.yaml (GNSS-disabled variant)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/config/pose_initializer.param.yaml`:

```yaml
/**:
  ros__parameters:
    user_defined_initial_pose:
      enable: false
      pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    gnss_pose_timeout: 3.0
    stop_check_duration: 3.0
    pose_error_threshold: 5.0
    pose_error_check_enabled: false
    ekf_enabled: true
    gnss_enabled: false
    yabloc_enabled: false
    ndt_enabled: true
    stop_check_enabled: true

    map_height_fitter:
      map_loader_name: "/map/pointcloud_map_loader"
      target: "pointcloud_map"

    gnss_particle_covariance:
      [
        1.0, 0.0, 0.0,  0.0,  0.0,  0.0,
        0.0, 1.0, 0.0,  0.0,  0.0,  0.0,
        0.0, 0.0, 0.01, 0.0,  0.0,  0.0,
        0.0, 0.0, 0.0,  0.01, 0.0,  0.0,
        0.0, 0.0, 0.0,  0.0,  0.01, 0.0,
        0.0, 0.0, 0.0,  0.0,  0.0,  10.0,
      ]

    output_pose_covariance:
      [
        1.0, 0.0, 0.0,  0.0,  0.0,  0.0,
        0.0, 1.0, 0.0,  0.0,  0.0,  0.0,
        0.0, 0.0, 0.01, 0.0,  0.0,  0.0,
        0.0, 0.0, 0.0,  0.01, 0.0,  0.0,
        0.0, 0.0, 0.0,  0.0,  0.01, 0.0,
        0.0, 0.0, 0.0,  0.0,  0.0,  0.2,
      ]
```

- [ ] **Step 2: Create localization.launch.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/launch/localization.launch.xml`:

```xml
<?xml version="1.0"?>
<launch>
  <arg name="map_path" default="$(env HOME)/maps/kachaka_home"/>

  <!-- Map -->
  <include file="$(find-pkg-share autoware_core_map)/launch/autoware_core_map.launch.xml">
    <arg name="lanelet2_map_path" value="$(var map_path)/lanelet2_map.osm"/>
    <arg name="map_projector_info_path" value="$(var map_path)/map_projector_info.yaml"/>
    <arg name="pointcloud_map_path" value="$(var map_path)/pointcloud_map.pcd"/>
    <arg name="pointcloud_map_metadata_path" value="$(var map_path)/pointcloud_map/metadata.yaml"/>
  </include>

  <!-- Localization with overridden pose_initializer config -->
  <include file="$(find-pkg-share autoware_core_localization)/launch/autoware_core_localization.launch.xml">
    <arg name="pose_initializer_param_path" value="$(find-pkg-share kachaka_autoware_bridge)/config/pose_initializer.param.yaml"/>
    <arg name="lidar_input_topic" value="/sensing/lidar/top/pointcloud_raw_ex"/>
    <arg name="vehicle_twist_input_topic" value="/sensing/vehicle_velocity_converter/twist_with_covariance"/>
    <arg name="gnss_input_topic" value=""/>
  </include>
</launch>
```

- [ ] **Step 3: Confirm the build**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

Expected result: build succeeds.

- [ ] **Step 4: Commit (runtime verification follows in Tasks 15-19)**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/launch/localization.launch.xml \
        ros2/kachaka_autoware_bridge/config/pose_initializer.param.yaml
git commit -m "$(cat <<'EOF'
feat(bridge): add localization launch with GNSS disabled

Wraps autoware_core_localization with pose_initializer config that
disables GNSS / YabLoc and keeps NDT + EKF + stop_check enabled.
Indoor / shelf-mounted-OS1 configuration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Temporary VelocityReport publisher for static vehicle_velocity_converter integration

**Purpose:** `/vehicle/status/velocity_status` is published by the Vehicle Interface node (Task 20+), but to verify Localization standalone first, a temporary relay converting Kachaka's `wheel_odometry` directly to `VelocityReport` is needed. It is folded into the Vehicle Interface node from Task 20 onwards.

Approach: a temporary Python-script-equivalent that subscribes to `wheel_odometry` and publishes `VelocityReport`. The real implementation lands as C++ in Tasks 25-27.

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml` (temporary during M2, removed at the end of M3)

- [ ] **Step 1: Implement the launch using `topic_tools` `transform` (avoid creating a python node)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml`:

```xml
<?xml version="1.0"?>
<launch>
  <!--
    TEMPORARY: Republish Kachaka wheel_odometry as Autoware VelocityReport.
    Replaced by kachaka_autoware_vehicle_interface in M3 (Task 25-27).
    This launch should be removed once M3 is done.
  -->
  <node pkg="topic_tools" exec="transform" name="wheel_odom_to_velocity_status">
    <param name="input_topic" value="/kachaka/wheel_odometry/wheel_odometry"/>
    <param name="output_topic" value="/vehicle/status/velocity_status"/>
    <param name="output_type" value="autoware_vehicle_msgs/msg/VelocityReport"/>
    <param name="expression" value="autoware_vehicle_msgs.msg.VelocityReport(header=m.header, longitudinal_velocity=m.twist.twist.linear.x, lateral_velocity=0.0, heading_rate=m.twist.twist.angular.z)"/>
    <param name="import" value="['autoware_vehicle_msgs.msg']"/>
  </node>
</launch>
```

Note: `topic_tools transform` is launchable from launch files on ROS 2 Jazzy. `expression` is a Python evaluation expression.

- [ ] **Step 2: Confirm the build**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

Expected result: build succeeds.

- [ ] **Step 3: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml
git commit -m "$(cat <<'EOF'
feat(bridge): temporary wheel_odom to VelocityReport relay for M2 testing

Uses topic_tools transform to bridge Kachaka wheel_odometry into the
Autoware /vehicle/status/velocity_status format. Replaced by the
vehicle_interface node in M3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Localization on real hardware + NDT Monte Carlo check

**Purpose:** With Kachaka stationary, confirm NDT settles on an initial pose and `/localization/kinematic_state` is published.

- [ ] **Step 1: Integrated launch across four terminals**

Terminal 1: Kachaka bridge
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka
```

Terminal 2: OS-1
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge sensor_ouster.launch.xml sensor_hostname:=<actual hostname>
```

Terminal 3: URDF + temporary velocity relay + localization
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_description robot_description.launch.py &
ros2 launch kachaka_autoware_bridge temp_velocity_relay.launch.xml &
ros2 launch autoware_core_sensing autoware_core_sensing.launch.xml &
ros2 launch kachaka_autoware_bridge localization.launch.xml \
  map_path:=$HOME/maps/kachaka_home
```

Terminal 4: RViz (can run on the development PC)
```bash
rviz2
```

- [ ] **Step 2: Provide a rough initial pose via `2D Pose Estimate` (near the M0 lanelet2 origin)**

In RViz, click `2D Pose Estimate` and drop an arrow on the map at "where Kachaka actually is".

Expected result: NDT converges via Monte Carlo and `/localization/kinematic_state` publishes at about 50 Hz.

- [ ] **Step 3: Inspect the output**

Run:
```bash
ros2 topic hz /localization/kinematic_state
ros2 topic echo --once /localization/kinematic_state
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo map base_footprint
```

Expected result:
- About 50 Hz.
- `pose.position` has plausible values.
- `map -> odom -> base_footprint` TF is consistent.

- [ ] **Step 4: Push Kachaka by hand for about 1 m and confirm NDT tracks**

Manual task: push Kachaka by hand for about 1 m.

Checks:
- The pointcloud and URDF in RViz move on the map.
- `/localization/kinematic_state` `pose.position` changes continuously.

Expected result: tracking holds, no divergence.

- [ ] **Step 5: Sanity-check wheel_odometry**

Run:
```bash
ros2 topic echo --once /kachaka/wheel_odometry/wheel_odometry
ros2 topic echo --once /vehicle/status/velocity_status
```

Check: when pushing Kachaka at about 0.1 m/s, does `longitudinal_velocity` read close to 0.1?

Expected result: plausible values (the verification point flagged in spec §6.1). If not, implement the IMU fallback in Task 17.

- [ ] **Step 6: Append verification notes (on success)**

Optionally record the verification result in `docs/superpowers/specs/2026-05-02-kachaka-autoware-core-design.md` §16.

Skip the commit if there is nothing to write.

---

### Task 17: IMU fallback when wheel_odometry is unreliable (conditional)

**Purpose:** Implement only if Task 16 Step 5 shows that `wheel_odometry` is unreliable. Addresses the verification point in spec §6.1.

**Precondition:** Skip this task entirely if wheel_odometry passed in Task 16.

**Files (only when implemented):**
- Modify: `ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml` (switch to IMU angular velocity)

Implementation approach: use IMU `/kachaka/imu/imu` `angular_velocity.z` as `heading_rate` and only use the `linear.x` component of wheel_odometry as `longitudinal_velocity`. Mechanically the same `topic_tools transform` as Task 15, just with a different expression.

- [ ] **Step 1: Decide whether this is needed (manual)**

Run the steps below if Task 16 Step 5 marked `wheel_odometry` as bad.

- [ ] **Step 2: IMU fallback launch (only when implementing)**

Skipped if Task 16 was OK. If implemented, combining two topics with `topic_tools` `transform` is awkward, so handle it inside `kachaka_autoware_vehicle_interface` instead (add a flag in Task 25).

- [ ] **Step 3: Commit (whether you skipped or implemented)**

Implemented: update the temp launch + commit. Skipped: do nothing.

---

### Task 18: M2 Localization verification complete

**Purpose:** Confirm the M2 exit criteria: NDT + EKF publish `/localization/kinematic_state`.

- [ ] **Step 1: Checklist**

1. `/localization/kinematic_state` publishes at 50 Hz.
2. Pushing Kachaka by 1 m tracks correctly.
3. `wheel_odometry` flows into the EKF through `vehicle_velocity_converter`.
4. `map -> odom` TF is steady (no divergence).

- [ ] **Step 2: No commit**

Verification only.

---

### Task 19: M2 -> M3 hand-off (keep the temporary relay)

**Purpose:** Keep `temp_velocity_relay.launch.xml` until M3 ships the real implementation. It is removed in Task 27.

- [ ] **Step 1: Note**

`temp_velocity_relay.launch.xml` is removed in Task 27. Re-check at M3 completion.

- [ ] **Step 2: No commit**

---

### Task 20: control_to_twist_converter test-first (part 1: straight line)

**Purpose:** Write the tests first (TDD): the first test for the Control -> Twist conversion verifies that `v=0.2 m/s, delta=0` yields `linear.x=0.2, angular.z=0`.

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp` (empty so the test fails to compile)
- Create: `ros2/kachaka_autoware_vehicle_interface/test/test_control_to_twist_converter.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: Place an empty header (so the test fails with a compile error)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_

namespace kachaka_autoware_vehicle_interface {

// Forward declarations only — implementation comes in Task 21.

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_
```

- [ ] **Step 2: Write the failing test**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/test/test_control_to_twist_converter.cpp`:

```cpp
#include <gtest/gtest.h>

#include <autoware_control_msgs/msg/control.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp"

using kachaka_autoware_vehicle_interface::ControlToTwistConverter;
using kachaka_autoware_vehicle_interface::ControlToTwistParams;

namespace {

ControlToTwistParams make_default_params()
{
  ControlToTwistParams p;
  p.wheel_base = 0.30;
  p.max_linear_velocity = 0.3;
  p.max_angular_velocity = 1.57;
  return p;
}

autoware_control_msgs::msg::Control make_control(double v, double delta)
{
  autoware_control_msgs::msg::Control c;
  c.longitudinal.velocity = static_cast<float>(v);
  c.lateral.steering_tire_angle = static_cast<float>(delta);
  return c;
}

}  // namespace

TEST(ControlToTwistConverter, StraightLineHasZeroAngular)
{
  ControlToTwistConverter converter(make_default_params());
  const auto twist = converter.convert(make_control(0.2, 0.0));
  EXPECT_DOUBLE_EQ(twist.linear.x, 0.2);
  EXPECT_DOUBLE_EQ(twist.angular.z, 0.0);
}
```

- [ ] **Step 3: Register the test in CMakeLists.txt**

Replace `/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt` with:

```cmake
cmake_minimum_required(VERSION 3.14)
project(kachaka_autoware_vehicle_interface)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_control_to_twist_converter
    test/test_control_to_twist_converter.cpp
  )
  target_include_directories(test_control_to_twist_converter PRIVATE include)
  ament_target_dependencies(test_control_to_twist_converter
    autoware_control_msgs
    geometry_msgs
  )
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
```

- [ ] **Step 4: Build the test and confirm it fails**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

Expected result: compile error (`ControlToTwistConverter` undefined). This confirms the failing test.

- [ ] **Step 5: No commit (do not commit a failing test)**

Commit after Task 21 implements the converter and the test passes.

---

### Task 21: Minimal control_to_twist_converter implementation that passes Task 20

**Purpose:** Write the minimal implementation that makes the Task 20 test pass.

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/control_to_twist_converter.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: Class definition in the header**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_

#include <autoware_control_msgs/msg/control.hpp>
#include <geometry_msgs/msg/twist.hpp>

namespace kachaka_autoware_vehicle_interface {

struct ControlToTwistParams
{
  double wheel_base;
  double max_linear_velocity;
  double max_angular_velocity;
};

class ControlToTwistConverter
{
public:
  explicit ControlToTwistConverter(const ControlToTwistParams & params);

  geometry_msgs::msg::Twist convert(const autoware_control_msgs::msg::Control & control) const;

private:
  ControlToTwistParams params_;
};

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_
```

- [ ] **Step 2: Implementation**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/src/control_to_twist_converter.cpp`:

```cpp
#include "kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp"

#include <algorithm>
#include <cmath>

namespace kachaka_autoware_vehicle_interface {

ControlToTwistConverter::ControlToTwistConverter(const ControlToTwistParams & params)
: params_(params) {}

geometry_msgs::msg::Twist ControlToTwistConverter::convert(
  const autoware_control_msgs::msg::Control & control) const
{
  const double v = static_cast<double>(control.longitudinal.velocity);
  const double delta = static_cast<double>(control.lateral.steering_tire_angle);
  const double omega = (params_.wheel_base > 0.0) ? v * std::tan(delta) / params_.wheel_base : 0.0;

  geometry_msgs::msg::Twist twist;
  twist.linear.x = std::clamp(v, -params_.max_linear_velocity, params_.max_linear_velocity);
  twist.angular.z = std::clamp(omega, -params_.max_angular_velocity, params_.max_angular_velocity);
  return twist;
}

}  // namespace kachaka_autoware_vehicle_interface
```

- [ ] **Step 3: Add library definition to CMakeLists.txt**

Update `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(kachaka_autoware_vehicle_interface)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/control_to_twist_converter.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_control_to_twist_converter
    test/test_control_to_twist_converter.cpp
  )
  target_link_libraries(test_control_to_twist_converter ${PROJECT_NAME})
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
```

- [ ] **Step 4: Build and run tests**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
colcon test-result --verbose --test-result-base build/kachaka_autoware_vehicle_interface
```

Expected result: 1 test in `test_control_to_twist_converter` passes.

- [ ] **Step 5: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): add ControlToTwistConverter (straight-line case)

Bicycle-to-differential-drive conversion. First gtest covers v=0.2,
delta=0 → linear.x=0.2, angular.z=0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 22: Additional control_to_twist_converter tests (curve / clamping / wheel_base=0)

**Purpose:** Cover boundary values and edge cases.

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/test/test_control_to_twist_converter.cpp`

- [ ] **Step 1: Add failing tests**

Append to `test_control_to_twist_converter.cpp`:

```cpp
TEST(ControlToTwistConverter, RightTurnHasNegativeAngular)
{
  // delta = -0.3 rad (right turn), v = 0.2 m/s, wheel_base = 0.30
  // omega = 0.2 * tan(-0.3) / 0.30 = -0.2061 rad/s
  ControlToTwistConverter converter(make_default_params());
  const auto twist = converter.convert(make_control(0.2, -0.3));
  EXPECT_NEAR(twist.linear.x, 0.2, 1e-6);
  EXPECT_NEAR(twist.angular.z, 0.2 * std::tan(-0.3) / 0.30, 1e-6);
}

TEST(ControlToTwistConverter, ClampLinearVelocityToMax)
{
  ControlToTwistConverter converter(make_default_params());
  const auto twist = converter.convert(make_control(1.0, 0.0));
  EXPECT_NEAR(twist.linear.x, 0.3, 1e-9);  // clamped to max_linear_velocity
}

TEST(ControlToTwistConverter, ClampAngularVelocityToMax)
{
  ControlToTwistConverter converter(make_default_params());
  // Large delta would yield huge omega; should be clamped to max_angular_velocity
  const auto twist = converter.convert(make_control(0.3, 1.5));
  EXPECT_NEAR(twist.angular.z, 1.57, 1e-6);
}

TEST(ControlToTwistConverter, NegativeLinearVelocityClampsAtNegativeMax)
{
  ControlToTwistConverter converter(make_default_params());
  const auto twist = converter.convert(make_control(-1.0, 0.0));
  EXPECT_NEAR(twist.linear.x, -0.3, 1e-9);
}

TEST(ControlToTwistConverter, ZeroWheelBaseGivesZeroAngular)
{
  // Defensive: wheel_base = 0 should not blow up.
  ControlToTwistParams p;
  p.wheel_base = 0.0;
  p.max_linear_velocity = 0.3;
  p.max_angular_velocity = 1.57;
  ControlToTwistConverter converter(p);
  const auto twist = converter.convert(make_control(0.2, 0.5));
  EXPECT_DOUBLE_EQ(twist.angular.z, 0.0);
}
```

Also add `<cmath>` to the includes at the top of the file:

```cpp
#include <cmath>
```

- [ ] **Step 2: Run the tests (all old + new tests must pass)**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
```

Expected result: all 5 tests pass.

- [ ] **Step 3: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/test/test_control_to_twist_converter.cpp
git commit -m "$(cat <<'EOF'
test(vehicle_interface): cover ControlToTwistConverter edge cases

Right turn, linear/angular clamping, zero wheel_base defensive case.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 23: operation_mode_state_machine test-first

**Purpose:** Write the tests first (TDD) for the STOP / AUTONOMOUS state transition logic.

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/test/test_operation_mode_state_machine.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: Empty header**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_

namespace kachaka_autoware_vehicle_interface {

// Defined in Task 24.

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_
```

- [ ] **Step 2: Failing tests (4 cases)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/test/test_operation_mode_state_machine.cpp`:

```cpp
#include <gtest/gtest.h>

#include "kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp"

using kachaka_autoware_vehicle_interface::OperationModeStateMachine;
using kachaka_autoware_vehicle_interface::OperationMode;

TEST(OperationModeStateMachine, InitialStateIsStop)
{
  OperationModeStateMachine sm;
  EXPECT_EQ(sm.get_state(), OperationMode::STOP);
}

TEST(OperationModeStateMachine, RequestAutonomousFromStopSucceeds)
{
  OperationModeStateMachine sm;
  EXPECT_TRUE(sm.request_autonomous());
  EXPECT_EQ(sm.get_state(), OperationMode::AUTONOMOUS);
}

TEST(OperationModeStateMachine, RequestStopFromAutonomousSucceeds)
{
  OperationModeStateMachine sm;
  sm.request_autonomous();
  EXPECT_TRUE(sm.request_stop());
  EXPECT_EQ(sm.get_state(), OperationMode::STOP);
}

TEST(OperationModeStateMachine, RequestSameStateIsIdempotent)
{
  OperationModeStateMachine sm;
  EXPECT_TRUE(sm.request_stop());
  EXPECT_EQ(sm.get_state(), OperationMode::STOP);
  sm.request_autonomous();
  EXPECT_TRUE(sm.request_autonomous());
  EXPECT_EQ(sm.get_state(), OperationMode::AUTONOMOUS);
}
```

- [ ] **Step 3: Add the test in CMakeLists.txt**

Append inside `if(BUILD_TESTING)` in CMakeLists.txt:

```cmake
  ament_add_gtest(test_operation_mode_state_machine
    test/test_operation_mode_state_machine.cpp
  )
  target_link_libraries(test_operation_mode_state_machine ${PROJECT_NAME})
```

- [ ] **Step 4: Confirm the build fails**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

Expected result: compile error (`OperationModeStateMachine` undefined).

- [ ] **Step 5: No commit (implementation lands in the next task)**

---

### Task 24: operation_mode_state_machine implementation

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/operation_mode_state_machine.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: Header**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_

#include <mutex>

namespace kachaka_autoware_vehicle_interface {

enum class OperationMode {
  STOP,
  AUTONOMOUS,
};

class OperationModeStateMachine
{
public:
  OperationModeStateMachine();

  OperationMode get_state() const;

  // Returns true on success. MVP scope: both transitions always succeed.
  bool request_autonomous();
  bool request_stop();

private:
  mutable std::mutex mutex_;
  OperationMode state_;
};

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_
```

- [ ] **Step 2: Implementation**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/src/operation_mode_state_machine.cpp`:

```cpp
#include "kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp"

namespace kachaka_autoware_vehicle_interface {

OperationModeStateMachine::OperationModeStateMachine()
: state_(OperationMode::STOP) {}

OperationMode OperationModeStateMachine::get_state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

bool OperationModeStateMachine::request_autonomous()
{
  std::lock_guard<std::mutex> lock(mutex_);
  state_ = OperationMode::AUTONOMOUS;
  return true;
}

bool OperationModeStateMachine::request_stop()
{
  std::lock_guard<std::mutex> lock(mutex_);
  state_ = OperationMode::STOP;
  return true;
}

}  // namespace kachaka_autoware_vehicle_interface
```

- [ ] **Step 3: Append to the library sources in CMakeLists.txt**

Append `src/operation_mode_state_machine.cpp` to the source list of `ament_auto_add_library`:

```cmake
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/control_to_twist_converter.cpp
  src/operation_mode_state_machine.cpp
)
```

- [ ] **Step 4: Run the tests**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
```

Expected result: 9 tests pass (5 + 4).

- [ ] **Step 5: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): add OperationModeStateMachine

Simple STOP/AUTONOMOUS state machine standing in for the Universe
autoware_command_mode_decider until C-scope expansion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 25: velocity_status_publisher test-first and implementation

**Purpose:** Implement the `nav_msgs/Odometry` -> `autoware_vehicle_msgs/VelocityReport` conversion.

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/velocity_status_publisher.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/velocity_status_publisher.cpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/test/test_velocity_status_publisher.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: Failing test**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/test/test_velocity_status_publisher.cpp`:

```cpp
#include <gtest/gtest.h>

#include <nav_msgs/msg/odometry.hpp>

#include "kachaka_autoware_vehicle_interface/velocity_status_publisher.hpp"

using kachaka_autoware_vehicle_interface::convert_odometry_to_velocity_report;

TEST(VelocityStatusPublisher, ConvertsLinearAndAngularComponents)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp.sec = 42;
  odom.header.stamp.nanosec = 123;
  odom.twist.twist.linear.x = 0.15;
  odom.twist.twist.angular.z = -0.5;

  const auto report = convert_odometry_to_velocity_report(odom);
  EXPECT_EQ(report.header.stamp.sec, 42);
  EXPECT_EQ(report.header.stamp.nanosec, 123u);
  EXPECT_FLOAT_EQ(report.longitudinal_velocity, 0.15f);
  EXPECT_FLOAT_EQ(report.lateral_velocity, 0.0f);
  EXPECT_FLOAT_EQ(report.heading_rate, -0.5f);
}
```

- [ ] **Step 2: Header (minimal declaration; will fail if test runs first)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/velocity_status_publisher.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VELOCITY_STATUS_PUBLISHER_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VELOCITY_STATUS_PUBLISHER_HPP_

#include <autoware_vehicle_msgs/msg/velocity_report.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace kachaka_autoware_vehicle_interface {

autoware_vehicle_msgs::msg::VelocityReport convert_odometry_to_velocity_report(
  const nav_msgs::msg::Odometry & odom);

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VELOCITY_STATUS_PUBLISHER_HPP_
```

- [ ] **Step 3: Implementation**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/src/velocity_status_publisher.cpp`:

```cpp
#include "kachaka_autoware_vehicle_interface/velocity_status_publisher.hpp"

namespace kachaka_autoware_vehicle_interface {

autoware_vehicle_msgs::msg::VelocityReport convert_odometry_to_velocity_report(
  const nav_msgs::msg::Odometry & odom)
{
  autoware_vehicle_msgs::msg::VelocityReport report;
  report.header = odom.header;
  report.longitudinal_velocity = static_cast<float>(odom.twist.twist.linear.x);
  report.lateral_velocity = 0.0f;  // differential-drive: lateral velocity is always zero in body frame
  report.heading_rate = static_cast<float>(odom.twist.twist.angular.z);
  return report;
}

}  // namespace kachaka_autoware_vehicle_interface
```

- [ ] **Step 4: Update CMakeLists.txt**

Append to the library sources:
```cmake
  src/velocity_status_publisher.cpp
```

Append the test inside `if(BUILD_TESTING)`:
```cmake
  ament_add_gtest(test_velocity_status_publisher
    test/test_velocity_status_publisher.cpp
  )
  target_link_libraries(test_velocity_status_publisher ${PROJECT_NAME})
```

- [ ] **Step 5: Build and test**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
```

Expected result: 10 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): add convert_odometry_to_velocity_report

Pure conversion from nav_msgs/Odometry (Kachaka wheel_odometry) to
autoware_vehicle_msgs/VelocityReport with lateral fixed at 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 26: Create vehicle_interface_node (minimal startup)

**Purpose:** Create the `rclcpp::Node` subclass with just enough wiring to start. Subscribers/publishers are wired in later tasks.

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/main.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_vehicle_interface/config/vehicle_interface.param.yaml`
- Create: `ros2/kachaka_autoware_vehicle_interface/launch/vehicle_interface.launch.xml`

- [ ] **Step 1: Header**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp"
#include "kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp"

namespace kachaka_autoware_vehicle_interface {

class VehicleInterfaceNode : public rclcpp::Node
{
public:
  explicit VehicleInterfaceNode(const rclcpp::NodeOptions & options);

private:
  std::unique_ptr<ControlToTwistConverter> converter_;
  OperationModeStateMachine state_machine_;
};

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_
```

- [ ] **Step 2: Implementation (minimal: parameter loading only)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`:

```cpp
#include "kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp"

namespace kachaka_autoware_vehicle_interface {

VehicleInterfaceNode::VehicleInterfaceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("kachaka_autoware_vehicle_interface", options)
{
  ControlToTwistParams params;
  params.wheel_base = declare_parameter<double>("wheel_base", 0.30);
  params.max_linear_velocity = declare_parameter<double>("max_linear_velocity", 0.3);
  params.max_angular_velocity = declare_parameter<double>("max_angular_velocity", 1.57);
  converter_ = std::make_unique<ControlToTwistConverter>(params);

  RCLCPP_INFO(
    get_logger(),
    "VehicleInterfaceNode started: wheel_base=%.3f, vmax=%.3f, wmax=%.3f",
    params.wheel_base, params.max_linear_velocity, params.max_angular_velocity);
}

}  // namespace kachaka_autoware_vehicle_interface
```

- [ ] **Step 3: main.cpp**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/src/main.cpp`:

```cpp
#include <rclcpp/rclcpp.hpp>

#include "kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<
    kachaka_autoware_vehicle_interface::VehicleInterfaceNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
```

- [ ] **Step 4: Add library sources and executable to CMakeLists.txt**

```cmake
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/control_to_twist_converter.cpp
  src/operation_mode_state_machine.cpp
  src/velocity_status_publisher.cpp
  src/vehicle_interface_node.cpp
)

ament_auto_add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})
```

- [ ] **Step 5: Parameter YAML (spec §9.2)**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/config/vehicle_interface.param.yaml`:

```yaml
/**:
  ros__parameters:
    max_linear_velocity: 0.3
    max_angular_velocity: 1.57
    wheel_base: 0.30
    cmd_vel_timeout: 0.5
    publish_period_velocity_status: 0.02
    publish_period_operation_mode: 0.1
    auto_enable_manual_control: true
```

- [ ] **Step 6: Launch file**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/launch/vehicle_interface.launch.xml`:

```xml
<?xml version="1.0"?>
<launch>
  <arg name="config_file" default="$(find-pkg-share kachaka_autoware_vehicle_interface)/config/vehicle_interface.param.yaml"/>

  <node pkg="kachaka_autoware_vehicle_interface"
        exec="kachaka_autoware_vehicle_interface_node"
        name="kachaka_autoware_vehicle_interface"
        output="screen">
    <param from="$(var config_file)"/>
  </node>
</launch>
```

- [ ] **Step 7: Build and smoke-test**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
source install/setup.bash
ros2 launch kachaka_autoware_vehicle_interface vehicle_interface.launch.xml &
sleep 2
ros2 node info /kachaka_autoware_vehicle_interface
kill %1
```

Expected result: the node starts and logs `VehicleInterfaceNode started: wheel_base=0.300, vmax=0.300, wmax=1.570`.

- [ ] **Step 8: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): scaffold VehicleInterfaceNode and launch

Loads ControlToTwistConverter parameters; subscriber/publisher wiring
follows in subsequent tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 27: Control subscriber + Twist publisher + operation_mode gate

**Purpose:** Subscribe to `/control/command/control_cmd`, watch `/system/operation_mode/state`, and republish to `/kachaka/manual_control/cmd_vel` only while AUTONOMOUS.

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`

- [ ] **Step 1: Add members to the header**

Updated `vehicle_interface_node.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <autoware_adapi_v1_msgs/msg/operation_mode_state.hpp>
#include <autoware_control_msgs/msg/control.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp"
#include "kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp"

namespace kachaka_autoware_vehicle_interface {

class VehicleInterfaceNode : public rclcpp::Node
{
public:
  explicit VehicleInterfaceNode(const rclcpp::NodeOptions & options);

private:
  std::unique_ptr<ControlToTwistConverter> converter_;
  OperationModeStateMachine state_machine_;

  rclcpp::Subscription<autoware_control_msgs::msg::Control>::SharedPtr control_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Subscription<autoware_adapi_v1_msgs::msg::OperationModeState>::SharedPtr op_mode_sub_;

  rclcpp::Time last_control_stamp_;
  double cmd_vel_timeout_sec_;

  void on_control(const autoware_control_msgs::msg::Control::SharedPtr msg);
  void on_op_mode_state(const autoware_adapi_v1_msgs::msg::OperationModeState::SharedPtr msg);
};

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_
```

- [ ] **Step 2: Update the implementation**

`vehicle_interface_node.cpp`:

```cpp
#include "kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp"

namespace kachaka_autoware_vehicle_interface {

VehicleInterfaceNode::VehicleInterfaceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("kachaka_autoware_vehicle_interface", options),
  last_control_stamp_(0, 0, RCL_ROS_TIME)
{
  ControlToTwistParams params;
  params.wheel_base = declare_parameter<double>("wheel_base", 0.30);
  params.max_linear_velocity = declare_parameter<double>("max_linear_velocity", 0.3);
  params.max_angular_velocity = declare_parameter<double>("max_angular_velocity", 1.57);
  cmd_vel_timeout_sec_ = declare_parameter<double>("cmd_vel_timeout", 0.5);
  converter_ = std::make_unique<ControlToTwistConverter>(params);

  control_sub_ = create_subscription<autoware_control_msgs::msg::Control>(
    "/control/command/control_cmd", rclcpp::QoS(1).transient_local(),
    std::bind(&VehicleInterfaceNode::on_control, this, std::placeholders::_1));

  twist_pub_ = create_publisher<geometry_msgs::msg::Twist>(
    "/kachaka/manual_control/cmd_vel", rclcpp::SensorDataQoS());

  op_mode_sub_ = create_subscription<autoware_adapi_v1_msgs::msg::OperationModeState>(
    "/system/operation_mode/state", rclcpp::QoS(1).transient_local(),
    std::bind(&VehicleInterfaceNode::on_op_mode_state, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "VehicleInterfaceNode started");
}

void VehicleInterfaceNode::on_control(
  const autoware_control_msgs::msg::Control::SharedPtr msg)
{
  last_control_stamp_ = now();
  if (state_machine_.get_state() != OperationMode::AUTONOMOUS) {
    return;
  }
  twist_pub_->publish(converter_->convert(*msg));
}

void VehicleInterfaceNode::on_op_mode_state(
  const autoware_adapi_v1_msgs::msg::OperationModeState::SharedPtr msg)
{
  // External adapi may set state from EngageButton; mirror it locally so
  // the gate uses the latest value.
  using OperationModeState = autoware_adapi_v1_msgs::msg::OperationModeState;
  if (msg->mode == OperationModeState::AUTONOMOUS) {
    state_machine_.request_autonomous();
  } else {
    state_machine_.request_stop();
  }
}

}  // namespace kachaka_autoware_vehicle_interface
```

Note: spec §9.1.E says "the Vehicle Interface implements a simple state machine and **publishes** `/system/operation_mode/state`". This task only **subscribes** for now; the publishing duty is added incrementally in Task 28 (since the core variant of autoware_default_adapi lacks the operation_mode interfaces, the Vehicle Interface must handle both sides).

- [ ] **Step 3: Build and confirm existing tests still pass**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface
```

Expected result: existing 10 tests pass. No new tests yet.

- [ ] **Step 4: Remove the temporary temp_velocity_relay**

Run:
```bash
cd ~/src/kachaka-api
rm ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml
```

Reason: it is superseded by the real implementation in M3 (see the Task 19 note about the M2 hand-off).

However, the Vehicle Interface does not yet publish VelocityReport at this point, so deletion is deferred until **after Task 28**. Skip the deletion here and perform it in Task 28 instead.

-> Step 4 is **skipped here and performed in Task 28**.

- [ ] **Step 4 (revised): Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): wire control_cmd subscriber and gated cmd_vel

Subscribes /control/command/control_cmd, mirrors operation_mode/state,
and publishes /kachaka/manual_control/cmd_vel only while AUTONOMOUS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 28: VelocityReport publishing + operation_mode publishing + change_to services

**Purpose:** Add the following to the Vehicle Interface:
- Subscribe to `wheel_odometry` and publish `/vehicle/status/velocity_status` at 50 Hz.
- Publish `/system/operation_mode/state` itself at 10 Hz.
- Host `/system/operation_mode/change_to_autonomous` and `change_to_stop` services.
- When `auto_enable_manual_control` is true, call `set_manual_control_enabled(true)` at startup.

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/package.xml` (verify that `std_srvs` is already there)

- [ ] **Step 1: Update header**

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <autoware_adapi_v1_msgs/msg/operation_mode_state.hpp>
#include <autoware_adapi_v1_msgs/srv/change_operation_mode.hpp>
#include <autoware_control_msgs/msg/control.hpp>
#include <autoware_vehicle_msgs/msg/velocity_report.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp"
#include "kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp"
#include "kachaka_autoware_vehicle_interface/velocity_status_publisher.hpp"

namespace kachaka_autoware_vehicle_interface {

class VehicleInterfaceNode : public rclcpp::Node
{
public:
  explicit VehicleInterfaceNode(const rclcpp::NodeOptions & options);

private:
  std::unique_ptr<ControlToTwistConverter> converter_;
  OperationModeStateMachine state_machine_;

  // I/O
  rclcpp::Subscription<autoware_control_msgs::msg::Control>::SharedPtr control_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::VelocityReport>::SharedPtr velocity_status_pub_;
  rclcpp::Publisher<autoware_adapi_v1_msgs::msg::OperationModeState>::SharedPtr op_mode_pub_;
  rclcpp::Service<autoware_adapi_v1_msgs::srv::ChangeOperationMode>::SharedPtr change_to_autonomous_srv_;
  rclcpp::Service<autoware_adapi_v1_msgs::srv::ChangeOperationMode>::SharedPtr change_to_stop_srv_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr enable_manual_control_client_;

  rclcpp::TimerBase::SharedPtr velocity_status_timer_;
  rclcpp::TimerBase::SharedPtr op_mode_timer_;
  rclcpp::TimerBase::SharedPtr cmd_vel_timeout_timer_;

  // Latest received
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  rclcpp::Time last_control_stamp_;
  double cmd_vel_timeout_sec_;

  // Callbacks
  void on_control(const autoware_control_msgs::msg::Control::SharedPtr msg);
  void on_odom(const nav_msgs::msg::Odometry::SharedPtr msg);
  void on_velocity_status_timer();
  void on_op_mode_timer();
  void on_cmd_vel_timeout_timer();
  void on_change_to_autonomous(
    const autoware_adapi_v1_msgs::srv::ChangeOperationMode::Request::SharedPtr req,
    autoware_adapi_v1_msgs::srv::ChangeOperationMode::Response::SharedPtr resp);
  void on_change_to_stop(
    const autoware_adapi_v1_msgs::srv::ChangeOperationMode::Request::SharedPtr req,
    autoware_adapi_v1_msgs::srv::ChangeOperationMode::Response::SharedPtr resp);

  void enable_manual_control(bool enable);
};

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_VEHICLE_INTERFACE_NODE_HPP_
```

- [ ] **Step 2: Update implementation**

Replace `vehicle_interface_node.cpp` with:

```cpp
#include "kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp"

#include <chrono>
#include <utility>

namespace kachaka_autoware_vehicle_interface {

using namespace std::chrono_literals;
using OperationModeState = autoware_adapi_v1_msgs::msg::OperationModeState;

VehicleInterfaceNode::VehicleInterfaceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("kachaka_autoware_vehicle_interface", options),
  last_control_stamp_(0, 0, RCL_ROS_TIME)
{
  ControlToTwistParams params;
  params.wheel_base = declare_parameter<double>("wheel_base", 0.30);
  params.max_linear_velocity = declare_parameter<double>("max_linear_velocity", 0.3);
  params.max_angular_velocity = declare_parameter<double>("max_angular_velocity", 1.57);
  cmd_vel_timeout_sec_ = declare_parameter<double>("cmd_vel_timeout", 0.5);
  const double velocity_status_period =
    declare_parameter<double>("publish_period_velocity_status", 0.02);
  const double op_mode_period =
    declare_parameter<double>("publish_period_operation_mode", 0.1);
  const bool auto_enable = declare_parameter<bool>("auto_enable_manual_control", true);
  converter_ = std::make_unique<ControlToTwistConverter>(params);

  // Subs
  control_sub_ = create_subscription<autoware_control_msgs::msg::Control>(
    "/control/command/control_cmd", rclcpp::QoS(1).transient_local(),
    std::bind(&VehicleInterfaceNode::on_control, this, std::placeholders::_1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/kachaka/wheel_odometry/wheel_odometry", rclcpp::SensorDataQoS(),
    std::bind(&VehicleInterfaceNode::on_odom, this, std::placeholders::_1));

  // Pubs
  twist_pub_ = create_publisher<geometry_msgs::msg::Twist>(
    "/kachaka/manual_control/cmd_vel", rclcpp::SensorDataQoS());
  velocity_status_pub_ = create_publisher<autoware_vehicle_msgs::msg::VelocityReport>(
    "/vehicle/status/velocity_status", rclcpp::QoS(1));
  op_mode_pub_ = create_publisher<OperationModeState>(
    "/system/operation_mode/state", rclcpp::QoS(1).transient_local());

  // Services
  change_to_autonomous_srv_ =
    create_service<autoware_adapi_v1_msgs::srv::ChangeOperationMode>(
      "/system/operation_mode/change_to_autonomous",
      std::bind(&VehicleInterfaceNode::on_change_to_autonomous, this,
        std::placeholders::_1, std::placeholders::_2));
  change_to_stop_srv_ =
    create_service<autoware_adapi_v1_msgs::srv::ChangeOperationMode>(
      "/system/operation_mode/change_to_stop",
      std::bind(&VehicleInterfaceNode::on_change_to_stop, this,
        std::placeholders::_1, std::placeholders::_2));

  // Client
  enable_manual_control_client_ =
    create_client<std_srvs::srv::SetBool>("/kachaka/manual_control/set_enabled");

  // Timers
  velocity_status_timer_ = create_wall_timer(
    std::chrono::duration<double>(velocity_status_period),
    std::bind(&VehicleInterfaceNode::on_velocity_status_timer, this));
  op_mode_timer_ = create_wall_timer(
    std::chrono::duration<double>(op_mode_period),
    std::bind(&VehicleInterfaceNode::on_op_mode_timer, this));
  cmd_vel_timeout_timer_ = create_wall_timer(
    100ms, std::bind(&VehicleInterfaceNode::on_cmd_vel_timeout_timer, this));

  if (auto_enable) {
    enable_manual_control(true);
  }

  RCLCPP_INFO(get_logger(), "VehicleInterfaceNode started");
}

void VehicleInterfaceNode::on_control(
  const autoware_control_msgs::msg::Control::SharedPtr msg)
{
  last_control_stamp_ = now();
  if (state_machine_.get_state() != OperationMode::AUTONOMOUS) {
    return;
  }
  twist_pub_->publish(converter_->convert(*msg));
}

void VehicleInterfaceNode::on_odom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  latest_odom_ = msg;
}

void VehicleInterfaceNode::on_velocity_status_timer()
{
  if (!latest_odom_) return;
  velocity_status_pub_->publish(convert_odometry_to_velocity_report(*latest_odom_));
}

void VehicleInterfaceNode::on_op_mode_timer()
{
  OperationModeState state;
  state.stamp = now();
  state.mode = (state_machine_.get_state() == OperationMode::AUTONOMOUS)
                 ? OperationModeState::AUTONOMOUS
                 : OperationModeState::STOP;
  state.is_autoware_control_enabled = (state.mode == OperationModeState::AUTONOMOUS);
  state.is_in_transition = false;
  state.is_stop_mode_available = true;
  state.is_autonomous_mode_available = true;
  state.is_local_mode_available = false;
  state.is_remote_mode_available = false;
  op_mode_pub_->publish(state);
}

void VehicleInterfaceNode::on_cmd_vel_timeout_timer()
{
  if (state_machine_.get_state() != OperationMode::AUTONOMOUS) return;
  const auto elapsed = (now() - last_control_stamp_).seconds();
  if (elapsed > cmd_vel_timeout_sec_) {
    geometry_msgs::msg::Twist zero;
    twist_pub_->publish(zero);
  }
}

void VehicleInterfaceNode::on_change_to_autonomous(
  const autoware_adapi_v1_msgs::srv::ChangeOperationMode::Request::SharedPtr /*req*/,
  autoware_adapi_v1_msgs::srv::ChangeOperationMode::Response::SharedPtr resp)
{
  state_machine_.request_autonomous();
  resp->status.success = true;
}

void VehicleInterfaceNode::on_change_to_stop(
  const autoware_adapi_v1_msgs::srv::ChangeOperationMode::Request::SharedPtr /*req*/,
  autoware_adapi_v1_msgs::srv::ChangeOperationMode::Response::SharedPtr resp)
{
  state_machine_.request_stop();
  resp->status.success = true;
}

void VehicleInterfaceNode::enable_manual_control(bool enable)
{
  if (!enable_manual_control_client_->wait_for_service(2s)) {
    RCLCPP_WARN(get_logger(), "/kachaka/manual_control/set_enabled service not available");
    return;
  }
  auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
  req->data = enable;
  enable_manual_control_client_->async_send_request(req);
  RCLCPP_INFO(get_logger(), "Requested set_manual_control_enabled(%s)", enable ? "true" : "false");
}

}  // namespace kachaka_autoware_vehicle_interface
```

- [ ] **Step 3: Build**

Run:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

Expected result: build succeeds.

- [ ] **Step 4: Standalone smoke test**

Run:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_vehicle_interface vehicle_interface.launch.xml &
sleep 2
ros2 topic list | grep -E "(operation_mode|velocity_status|cmd_vel)"
ros2 service list | grep operation_mode
ros2 topic echo --once /system/operation_mode/state
kill %1
```

Expected result:
- `/system/operation_mode/state`, `/vehicle/status/velocity_status`, and `/kachaka/manual_control/cmd_vel` are visible.
- `change_to_autonomous` and `change_to_stop` services are visible.
- `OperationModeState` shows `mode: 1` (STOP).

- [ ] **Step 5: Remove the temporary temp_velocity_relay.launch.xml**

Run:
```bash
cd ~/src/kachaka-api
rm ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml
```

- [ ] **Step 6: Commit**

Run:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_vehicle_interface/ ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml
git commit -m "$(cat <<'EOF'
feat(vehicle_interface): full I/O wiring for VehicleInterfaceNode

- Publish /vehicle/status/velocity_status from Kachaka wheel_odometry
- Publish /system/operation_mode/state at 10 Hz
- Host /system/operation_mode/change_to_{autonomous,stop} services
- Auto-call /kachaka/manual_control/set_enabled(true) on startup
- Zero-Twist failsafe when control_cmd times out

Removes the temporary topic_tools relay introduced for M2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 29: M3 Vehicle Interface integration check on real hardware

**Purpose:** Confirm the Vehicle Interface behaves correctly together with Kachaka.

- [ ] **Step 1: Integrated startup**

Terminal A:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka
```

Terminal B:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_vehicle_interface vehicle_interface.launch.xml
```

- [ ] **Step 2: Verify automatic manual_control activation**

Run (in another terminal):
```bash
ros2 topic echo --once /vehicle/status/velocity_status
ros2 topic echo --once /system/operation_mode/state
```

Expected result: VelocityReport and OperationModeState publish. The Vehicle Interface startup log contains `Requested set_manual_control_enabled(true)`.

- [ ] **Step 3: Transition to AUTONOMOUS and send a synthetic Control**

Run:
```bash
# Switch to AUTONOMOUS
ros2 service call /system/operation_mode/change_to_autonomous \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}"

# Publish a synthetic Control once (v=0.1, delta=0)
ros2 topic pub --once /control/command/control_cmd \
  autoware_control_msgs/msg/Control \
  '{longitudinal: {velocity: 0.1, acceleration: 0.0}, lateral: {steering_tire_angle: 0.0}}'

# Confirm it was relayed
ros2 topic echo --once /kachaka/manual_control/cmd_vel
```

Expected result: `cmd_vel` shows `linear.x: 0.1, angular.z: 0.0`. Kachaka moves forward (or, if `set_manual_control_enabled` was already true, it moves at 0.1 m/s). **Safety note: lift Kachaka off the ground or run this in a collision-free area.**

- [ ] **Step 4: Return to STOP**

Run:
```bash
ros2 service call /system/operation_mode/change_to_stop \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}"
ros2 topic pub --once /control/command/control_cmd \
  autoware_control_msgs/msg/Control \
  '{longitudinal: {velocity: 0.1, acceleration: 0.0}, lateral: {steering_tire_angle: 0.0}}'
ros2 topic echo --once /kachaka/manual_control/cmd_vel
```

Expected result: while STOP, `cmd_vel` is not updated (or only zero Twist). Kachaka stays still.

- [ ] **Step 5: Commit — verification notes only (no code changes)**

Skip if there is nothing to record.

---

### Task 30: M3 exit checkpoint

**Purpose:** Confirm the M3 exit criteria: Control->Twist conversion, velocity_status, operation_mode state machine, automatic ManualControl enable.

- [ ] **Step 1: Checklist**

1. `colcon test` runs all gtests with PASS.
2. `vehicle_interface` calls `set_manual_control_enabled(true)` on startup.
3. `cmd_vel` only flows while AUTONOMOUS.
4. `/vehicle/status/velocity_status` publishes at 50 Hz.
5. `/system/operation_mode/state` publishes at 10 Hz.
6. `change_to_autonomous` and `change_to_stop` services respond.
7. After `cmd_vel_timeout`, a zero Twist is published.

- [ ] **Step 2: No commit**

Verification only.

---

### Task 31: AD-API launch wrapper

**Purpose:** Wrap `autoware_core_api.launch.xml` for the Kachaka integration.

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/api.launch.xml`

- [ ] **Step 1: api.launch.xml**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/launch/api.launch.xml`:

```xml
<?xml version="1.0"?>
<launch>
  <!-- Wraps autoware_core_api: default_adapi + adaptors -->
  <include file="$(find-pkg-share autoware_core_api)/launch/autoware_core_api.launch.xml">
    <arg name="launch_default_adapi" value="true"/>
    <arg name="launch_rviz_adaptors" value="true"/>
  </include>
</launch>
```

- [ ] **Step 2: Build**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

Expected result: build succeeds.

- [ ] **Step 3: Standalone smoke test**

```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge api.launch.xml &
sleep 3
ros2 service list | grep -E "/api/(localization|routing|operation_mode)"
kill %1
```

Expected result: services such as `/api/localization/initialize` and `/api/routing/set_route_points` are visible (`/api/operation_mode/*` is not provided by the autoware_core variant; that is fine).

- [ ] **Step 4: Commit**

```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/launch/api.launch.xml
git commit -m "$(cat <<'EOF'
feat(bridge): add api launch wrapping autoware_default_adapi

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 32: Planning launch wrapper

**Purpose:** Invoke `autoware_core_planning.launch.xml` with the vehicle_info from kachaka_autoware_description.

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/planning.launch.xml`

- [ ] **Step 1: planning.launch.xml**

```xml
<?xml version="1.0"?>
<launch>
  <arg name="vehicle_info_param_file"
       default="$(find-pkg-share kachaka_autoware_description)/config/vehicle_info.param.yaml"/>

  <include file="$(find-pkg-share autoware_core_planning)/launch/autoware_core_planning.launch.xml">
    <arg name="vehicle_param_file" value="$(var vehicle_info_param_file)"/>
    <!-- MVP: ObstacleStop disabled; enabled later in M6 -->
    <arg name="motion_velocity_planner_launch_modules" value="[]"/>
  </include>
</launch>
```

Note: `autoware_core_planning.launch.xml` may require a `vehicle_model` argument. If it errors, pass a dummy `vehicle_model` such as `kachaka` (only `vehicle_info_param_file` is actually consumed).

- [ ] **Step 2: Build and standalone smoke test**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
source install/setup.bash
ros2 launch kachaka_autoware_bridge planning.launch.xml \
  vehicle_info_param_file:=$(ros2 pkg prefix kachaka_autoware_description)/share/kachaka_autoware_description/config/vehicle_info.param.yaml &
sleep 5
ros2 node list | grep -E "(mission_planner|behavior_velocity|motion_velocity|velocity_smoother)"
kill %1
```

Expected result: the nodes start, though some remain in a wait state because map and localization are not running.

- [ ] **Step 3: Commit**

```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/launch/planning.launch.xml
git commit -m "$(cat <<'EOF'
feat(bridge): add planning launch with Kachaka vehicle_info

ObstacleStop module disabled in MVP; enabled in M6 with OS-1 derived
pointcloud.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 33: Control launch wrapper

**Purpose:** Launch `autoware_core_control.launch.xml` so `simple_pure_pursuit`'s output `/control/command/control_cmd` reaches the Vehicle Interface.

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/control.launch.xml`

- [ ] **Step 1: control.launch.xml**

```xml
<?xml version="1.0"?>
<launch>
  <arg name="vehicle_info_param_file"
       default="$(find-pkg-share kachaka_autoware_description)/config/vehicle_info.param.yaml"/>

  <include file="$(find-pkg-share autoware_core_control)/launch/autoware_core_control.launch.xml">
    <arg name="vehicle_info_param_file" value="$(var vehicle_info_param_file)"/>
  </include>
</launch>
```

- [ ] **Step 2: Build and standalone smoke test**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
source install/setup.bash
ros2 launch kachaka_autoware_bridge control.launch.xml &
sleep 3
ros2 node list | grep simple_pure_pursuit
kill %1
```

Expected result: the `simple_pure_pursuit` node starts.

- [ ] **Step 3: Commit**

```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/launch/control.launch.xml
git commit -m "$(cat <<'EOF'
feat(bridge): add control launch wrapping autoware_simple_pure_pursuit

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 34: Top-level launch — kachaka_autoware.launch.xml

**Purpose:** Bring up every component via a single launch file.

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/kachaka_autoware.launch.xml`

- [ ] **Step 1: Top-level launch**

```xml
<?xml version="1.0"?>
<launch>
  <arg name="server_uri" default="192.168.1.91:26400" description="Kachaka gRPC URI"/>
  <arg name="kachaka_namespace" default="kachaka"/>
  <arg name="frame_prefix" default=""/>
  <arg name="sensor_hostname" default="os-122000000000.local" description="Ouster OS-1 hostname"/>
  <arg name="map_path" default="$(env HOME)/maps/kachaka_home"/>

  <!-- 1. Kachaka gRPC bridge -->
  <include file="$(find-pkg-share kachaka_grpc_ros2_bridge)/launch/grpc_ros2_bridge.launch.xml">
    <arg name="server_uri" value="$(var server_uri)"/>
    <arg name="namespace" value="$(var kachaka_namespace)"/>
    <arg name="frame_prefix" value="$(var frame_prefix)"/>
  </include>

  <!-- 2. Robot description with shelf+OS-1 -->
  <include file="$(find-pkg-share kachaka_autoware_description)/launch/robot_description.launch.py"/>

  <!-- 3. Ouster sensor -->
  <include file="$(find-pkg-share kachaka_autoware_bridge)/launch/sensor_ouster.launch.xml">
    <arg name="sensor_hostname" value="$(var sensor_hostname)"/>
  </include>

  <!-- 4. Sensing (vehicle_velocity_converter) -->
  <include file="$(find-pkg-share autoware_core_sensing)/launch/autoware_core_sensing.launch.xml"/>

  <!-- 5. Localization (NDT + EKF + map) -->
  <include file="$(find-pkg-share kachaka_autoware_bridge)/launch/localization.launch.xml">
    <arg name="map_path" value="$(var map_path)"/>
  </include>

  <!-- 6. Planning -->
  <include file="$(find-pkg-share kachaka_autoware_bridge)/launch/planning.launch.xml"/>

  <!-- 7. Control -->
  <include file="$(find-pkg-share kachaka_autoware_bridge)/launch/control.launch.xml"/>

  <!-- 8. Vehicle Interface (Control → Twist + operation_mode) -->
  <include file="$(find-pkg-share kachaka_autoware_vehicle_interface)/launch/vehicle_interface.launch.xml"/>

  <!-- 9. AD-API -->
  <include file="$(find-pkg-share kachaka_autoware_bridge)/launch/api.launch.xml"/>
</launch>
```

Note: `autoware_core_sensing`'s `vehicle_velocity_converter` reads `vehicle/status/velocity_status`. Since the Vehicle Interface publishes it, starting `kachaka_autoware_vehicle_interface` first (or in parallel) is fine — DDS tolerates late starts.

- [ ] **Step 2: Build**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

Expected result: build succeeds.

- [ ] **Step 3: Commit (real-hardware verification follows in the next task)**

```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/launch/kachaka_autoware.launch.xml
git commit -m "$(cat <<'EOF'
feat(bridge): add top-level kachaka_autoware launch

Single entry point that brings up Kachaka gRPC bridge, OS-1, URDF,
sensing, localization, planning, control, vehicle_interface, and AD-API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 35: M4 Planning verification on real hardware + perception_stub decision

**Purpose:** Confirm Planning publishes `/planning/trajectory`. If perception inputs are required, add a `perception_stub`.

- [ ] **Step 1: Run the integrated launch**

```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge kachaka_autoware.launch.xml \
  server_uri:=192.168.1.91:26400 \
  sensor_hostname:=<actual hostname> \
  map_path:=$HOME/maps/kachaka_home
```

- [ ] **Step 2: Inspect node logs**

In another terminal:
```bash
ros2 node list | wc -l                           # many nodes should appear
ros2 topic list | grep -E "(planning|control|localization)" | head -20
```

Check: do `behavior_velocity_planner` or `motion_velocity_planner` log "topic not received" errors? If they do, a perception_stub is needed.

- [ ] **Step 3: Add perception_stub if required**

If errors complain about missing topics such as `dynamic_objects` (`autoware_perception_msgs/PredictedObjects`), `occupancy_grid_map`, or `traffic_signals`, create `perception_stub.launch.xml` that publishes empty messages at 1 Hz:

```xml
<?xml version="1.0"?>
<launch>
  <node pkg="topic_tools" exec="transform" name="empty_objects_pub">
    <param name="input_topic" value="/__null"/>
    <param name="output_topic" value="/perception/object_recognition/objects"/>
    <param name="output_type" value="autoware_perception_msgs/msg/PredictedObjects"/>
    <param name="expression" value="autoware_perception_msgs.msg.PredictedObjects()"/>
    <param name="import" value="['autoware_perception_msgs.msg']"/>
  </node>
  <!-- Add the same pattern for other topics as needed -->
</launch>
```

`topic_tools transform` **cannot publish from an empty input**, so when needed, create a small Python node `kachaka_autoware_bridge/scripts/perception_stub.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from autoware_perception_msgs.msg import PredictedObjects
# add other required types


class PerceptionStub(Node):
    def __init__(self):
        super().__init__("perception_stub")
        self.objects_pub = self.create_publisher(PredictedObjects, "/perception/object_recognition/objects", 1)
        self.timer = self.create_timer(1.0, self.publish_all)

    def publish_all(self):
        msg = PredictedObjects()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        self.objects_pub.publish(msg)


def main():
    rclpy.init()
    node = PerceptionStub()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

Also add the install rule to CMakeLists.txt:
```cmake
install(PROGRAMS scripts/perception_stub.py DESTINATION lib/${PROJECT_NAME})
```

- [ ] **Step 4: Commit perception_stub if implemented**

Only when implemented:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/scripts/ ros2/kachaka_autoware_bridge/CMakeLists.txt ros2/kachaka_autoware_bridge/launch/perception_stub.launch.xml
git commit -m "feat(bridge): add perception_stub for missing perception inputs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 36: Verify trajectory generation via manual set_route_points (M4 exit criteria)

**Purpose:** Hit AD-API `set_route_points` from the CLI and confirm `/planning/trajectory` publishes.

- [ ] **Step 1: With the integrated launch already running, set the initial pose**

```bash
ros2 service call /api/localization/initialize \
  autoware_adapi_v1_msgs/srv/InitializeLocalization \
  "{pose_with_covariance: [{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}, covariance: [0.25, 0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0.0,0,0, 0,0,0,0,0.0,0, 0,0,0,0,0,0.0698]}}]}"
```

Expected result: NDT settles on an initial pose via Monte Carlo.

- [ ] **Step 2: Send a single-point goal**

```bash
ros2 service call /api/routing/set_route_points \
  autoware_adapi_v1_msgs/srv/SetRoutePoints \
  "{header: {frame_id: 'map'}, goal: {position: {x: 1.5, y: 0.0, z: 0.0}, orientation: {w: 1.0}}, waypoints: [], option: {}}"
```

Expected result: a trajectory is generated.

- [ ] **Step 3: Verify the trajectory**

```bash
ros2 topic echo --once /planning/trajectory | head -30
ros2 topic hz /planning/trajectory
```

Expected result: the trajectory is published periodically.

- [ ] **Step 4: No commit (verification)**

---

### Task 37: M5 closed-loop run (MVP achieved)

**Purpose:** Confirm Kachaka reaches a goal placed once in RViz.

- [ ] **Step 1: Create the rviz config file**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/config/autoware.rviz`:

A minimal config with the autoware_rviz_plugins panels. The easiest path is to configure interactively in RViz and save:

```bash
rviz2
# Display: TF, RobotModel(/robot_description), PointCloud2(/sensing/lidar/top/pointcloud_raw_ex),
#          MarkerArray(/planning/scenario_planning/lane_driving/behavior_planning/path),
#          Path(/planning/trajectory)
# Panels: InitialPoseButtonPanel, RouteTool, EngageButton, AutowareStatePanel
# Save: File -> Save As -> autoware.rviz
```

Add the install rule in CMakeLists.txt:

```cmake
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
```

- [ ] **Step 2: Launch the integrated stack and RViz**

Terminal A (on Thor):
```bash
ros2 launch kachaka_autoware_bridge kachaka_autoware.launch.xml \
  server_uri:=192.168.1.91:26400 \
  sensor_hostname:=<actual hostname> \
  map_path:=$HOME/maps/kachaka_home
```

Terminal B (development PC):
```bash
rviz2 -d $(ros2 pkg prefix kachaka_autoware_bridge)/share/kachaka_autoware_bridge/config/autoware.rviz
```

- [ ] **Step 3: MVP operating sequence (spec §10.3)**

1. Click "Initialize" on RViz `InitialPoseButtonPanel` (or use `2D Pose Estimate` for a rough pose).
2. NDT converges and the robot is shown at the correct position on the map.
3. Use `RouteTool` (or the standard `2D Goal Pose`) to set a goal about 1.5 m ahead.
4. The trajectory is rendered.
5. Press `EngageButton`.
6. Kachaka starts moving, reaches the goal, and stops automatically.

- [ ] **Step 4: Success criteria check (spec §13.3)**

- Reach the goal within +/-0.3 m and +/-0.2 rad.
- No human intervention.

MVP is achieved when 5 different goals each succeed 80% of the time across 3 attempts.

- [ ] **Step 5: Tuning candidates on failure**

- `wheel_base` virtual value (vehicle_info.param.yaml) — increase if turning is too sharp, decrease if too sluggish.
- `simple_pure_pursuit` `lookahead_gain` / `lookahead_min_distance`.
- NDT `voxel_size` and iteration count.
- EKF Q/R covariances.

After tuning, edit the corresponding yaml and commit:
```bash
git add ros2/kachaka_autoware_description/config/vehicle_info.param.yaml
git commit -m "tune(description): adjust wheel_base virtual value to X.XX based on M5

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Verification complete -> release notes / README addendum**

Either rewrite the §16 risk row of `docs/superpowers/specs/2026-05-02-kachaka-autoware-core-design.md` to "M5 hardware tuning complete, wheel_base = X.XX", or create a new `kachaka_autoware_bridge/README.md` that records the MVP achievement.

```bash
git add ros2/kachaka_autoware_bridge/README.md
git commit -m "docs(bridge): MVP M5 closed-loop achieved

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

### Spec coverage

Map each spec section to the corresponding tasks:

| Spec section | Task |
|---|---|
| §3.3-1 pointcloud_map | Task 2 |
| §3.3-2 lanelet2 | Task 3 |
| §3.3-3 OS-1 physical mounting | Task 4 |
| §4.2 TF tree (docking_link -> shelf -> os1) | Task 6, 7 (URDF) |
| §4.3 Frame alignment | Task 3 (alignment via Vector Map Builder) |
| §5 kachaka_description enhancements | Task 6 |
| §5 New package layout | Task 5, 7, 10, 12 |
| §6 Localization | Task 14, 16 |
| §6.1 wheel_odometry verification | Task 16 Step 5, Task 17 |
| §7 Planning | Task 32 |
| §7.1 Goal-receiving flow | Task 31, 36 |
| §8 Control | Task 33 |
| §9.1.A Control->Twist | Task 20-22, 27 |
| §9.1.B cmd_vel gate | Task 27 |
| §9.1.C VelocityReport | Task 25, 28 |
| §9.1.D Automatic ManualControl | Task 28 |
| §9.1.E Operation Mode | Task 23-24, 28 |
| §9.2 Settings | Task 26 |
| §9.3 vehicle_info | Task 8 |
| §10 AD-API | Task 31 |
| §10.2 RViz panels | Task 1 (clone), 37 (RViz config) |
| §10.3 Operating sequence | Task 37 |
| §11 Data flow | Task 34 (top-level launch) |
| §12 Error handling / timeout | Task 28 (zero Twist failsafe) |
| §13.1 Unit tests | Task 20-25 |
| §13.3 System tests | Task 37 |
| §14 Milestones | Task 1-37 (all) |
| §16 Risk: wheel_base tuning | Task 37 Step 5 |
| §16 Risk: wheel_odometry | Task 16 Step 5 + Task 17 |

Gap analysis:
- §13.2 integration tests (rosbag regression): out of scope for MVP, planned for the post-M5 operations phase (matches the spec §2.2 follow-on framing). OK
- §16 risk NDT divergence: handled during M5 hardware tuning (Task 37 Step 5). OK
- §17 open questions: not required for MVP; defer. OK
- §16 obstacle stop (M6): out of scope for MVP (spec §2.2 OOS). Disabled in Task 32 via `motion_velocity_planner_launch_modules: []`. OK

### Placeholder scan

- "TBD" / "TODO" / "implement later": none. The only conditional implementation is Task 35's perception_stub, with explicit implement-if-needed criteria.
- "Add appropriate error handling": none (concrete timeout values and rates are stated).
- "Similar to Task N": none (Task 21-22 is fully expanded as separate tasks).
- Unknown types / functions: every type and function is defined in a preceding task (`ControlToTwistConverter` in Task 21, `OperationModeStateMachine` in Task 24, `convert_odometry_to_velocity_report` in Task 25). OK

### Type consistency

- `ControlToTwistConverter` / `ControlToTwistParams`: header declared in Task 20, implemented in Task 21. Consistent.
- `OperationMode` enum / `OperationModeStateMachine`: consistent across Task 23 and 24.
- `convert_odometry_to_velocity_report` (snake_case function): consistent in Task 25.
- `VehicleInterfaceNode`: incrementally extended in Task 26 -> 27 -> 28. Field and method names are consistent.
- launch names: `vehicle_interface.launch.xml`, `kachaka_autoware.launch.xml`, `localization.launch.xml`, `planning.launch.xml`, `control.launch.xml`, `api.launch.xml`, `sensor_ouster.launch.xml`. All names are consistent with their downstream references.

### Scope check

37 tasks cover spec milestones M0-M5. M6 is out of scope (matching the spec). Each task is independently implementable with clear exit criteria, and the work fits in a single implementation plan.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-kachaka-autoware-core-mvp.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Particularly effective for the TDD-style tasks in M3.

**2. Inline Execution** — batched execution in this session with pauses at checkpoints. Stalls when blocked on real-hardware work like M0.

**Which approach?**

This plan also contains many **physical / real-hardware / external-hardware-dependent tasks**:
- Task 2 (M0 mapping drive)
- Task 3 (Vector Map Builder web UI work)
- Task 4 (OS-1 physical mounting)
- Task 11, 16, 29, 35, 36, 37 (real-hardware verification)

These cannot be completed by an agent alone, so the recommended workflow is to **split the code-implementation tasks (Task 5-10, 12, 14-15, 20-28, 31-34) from the hardware tasks** and run them separately.
