# Kachaka × Autoware Core MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kachaka を Autoware Core のスタック（NDT localization / lanelet2 planning / simple_pure_pursuit control / AD-API）で自律移動させる。MVP は RViz の Autoware 標準UIで 2D Goal Pose を 1 点指定すると Kachaka が目標到達すること。

**Architecture:** kachaka-api リポジトリの `ros2/` 配下に 4 個の新規パッケージ（`kachaka_autoware_bridge`, `kachaka_autoware_vehicle_interface`, `kachaka_autoware_description`, `kachaka_autoware_maps`）を追加し、既存の `kachaka_grpc_ros2_bridge` と Autoware Core を Vehicle Interface ノード経由で繋ぐ。Vehicle Interface は `autoware_control_msgs/Control`（Ackermann）→ `geometry_msgs/Twist`（差動駆動）変換、`/vehicle/status/velocity_status` 発行、Operation Mode 簡易状態機械、ManualControl 自動有効化を担う。

**Tech Stack:** ROS 2 Jazzy / C++17 / `rclcpp` / `ament_cmake_auto` / gtest（`ament_cmake_gtest`）/ launch_xml / Autoware Core (autoware_core_localization / planning / control / api) / autoware_rviz_plugins / ouster-ros driver / Vector Map Builder（外部）/ lio_sam or fast_lio or glim（外部、M0で選択）

**前提となるユーザー側ハードウェア / 環境:**
- Kachaka 本体: 192.168.1.91、gRPC API port 26400、SW 3.16+
- Jetson Thor（Ubuntu 24.04 + ROS 2 Jazzy、ROS_DOMAIN_ID=123、ws=`~/ros/jazzy`）
- Ouster OS-1 128（シェルフに固定済 / 固定予定）
- 開発 PC（Jazzy + RViz2、同一 ROS_DOMAIN_ID）
- **Kachakaの 2D LiDAR は故障**（Kachaka 内蔵 SLAM・地図作成・自己位置推定を使用しない）

**前提となるリポジトリ:**
- `~/src/kachaka-api` (本リポジトリ)
- `~/src/autoware_core` (clone 済)
- `~/ros/jazzy/src/` 配下に他依存パッケージを clone

---

## File Structure

### 新規パッケージ（kachaka-api リポジトリ内）

```
ros2/
├── kachaka_autoware_bridge/                              # メタ + 統合 launch
│   ├── package.xml                                       # 依存: 他3パッケージ + autoware_core_*
│   ├── CMakeLists.txt
│   └── launch/
│       └── kachaka_autoware.launch.xml                   # 全部入りエントリ。bridge + Autoware Core 全部
│
├── kachaka_autoware_vehicle_interface/                   # ★ 中核
│   ├── package.xml                                       # 依存: rclcpp / autoware_control_msgs / autoware_vehicle_msgs / autoware_adapi_v1_msgs / geometry_msgs / nav_msgs / std_srvs
│   ├── CMakeLists.txt
│   ├── include/kachaka_autoware_vehicle_interface/
│   │   ├── control_to_twist_converter.hpp               # Control msg → Twist 変換ロジック（純粋関数）
│   │   ├── operation_mode_state_machine.hpp             # STOP / AUTONOMOUS の簡易 state machine
│   │   ├── velocity_status_publisher.hpp                # Odometry → VelocityReport 変換
│   │   └── vehicle_interface_node.hpp                   # rclcpp::Node サブクラス
│   ├── src/
│   │   ├── control_to_twist_converter.cpp
│   │   ├── operation_mode_state_machine.cpp
│   │   ├── velocity_status_publisher.cpp
│   │   ├── vehicle_interface_node.cpp                   # サブモジュールの組み立てと ROS I/O
│   │   └── main.cpp                                     # rclcpp::spin
│   ├── launch/
│   │   └── vehicle_interface.launch.xml
│   ├── config/
│   │   └── vehicle_interface.param.yaml
│   └── test/
│       ├── CMakeLists.txt                                # test 単独 CMake fragment
│       ├── test_control_to_twist_converter.cpp          # 純粋ロジックの境界値テスト
│       ├── test_operation_mode_state_machine.cpp        # 状態遷移テスト
│       └── test_velocity_status_publisher.cpp           # Odometry → VelocityReport 変換テスト
│
├── kachaka_autoware_description/                         # URDF + vehicle_info
│   ├── package.xml                                       # 依存: kachaka_description / xacro / ouster_description
│   ├── CMakeLists.txt
│   ├── urdf/
│   │   ├── kachaka_autoware.urdf.xacro                  # kachaka.urdf.xacro を include + シェルフ追加
│   │   └── shelf_with_ouster.urdf.xacro                 # base_link → shelf_dock_link → os1_sensor の static joint
│   ├── config/
│   │   └── vehicle_info.param.yaml                      # 差動駆動向け仮想値
│   └── launch/
│       └── robot_description.launch.py                  # robot_state_publisher 起動
│
└── kachaka_autoware_maps/                                # サンプル + 手順書
    ├── package.xml
    ├── CMakeLists.txt
    └── README.md                                         # M0 の手順書（user_dir/maps の指示）
```

### 各責務（split by responsibility, not technical layer）

- **`control_to_twist_converter`**: 純粋関数 `Twist convert(const Control&, params)`。ROS に依存しない。テストしやすさ最優先。
- **`operation_mode_state_machine`**: 純粋クラス `OperationModeStateMachine`。状態遷移ロジック、`get_state()`, `request_autonomous()`, `request_stop()`。`rclcpp::Node` 非依存。
- **`velocity_status_publisher`**: 純粋関数 `VelocityReport convert(const Odometry&)`。ROS msgs だが node 非依存。
- **`vehicle_interface_node`**: 上記サブモジュールを所有する `rclcpp::Node`。subscriber/publisher/service/timer の配線のみ。ロジックを含まない。

この分離により、サブモジュール単体で gtest でき、将来 `operation_mode_state_machine` を Universe の `autoware_command_mode_decider` に置き換える際は `vehicle_interface_node` の組み立てを差し替えるだけで済む。

### 外部 clone するパッケージ（既に手順は仕様書 §10.2）

- `autoware_rviz_plugins` を `~/ros/jazzy/src/autoware_rviz_plugins/` に clone
- `ouster-ros` を `~/ros/jazzy/src/ouster-ros/` に clone（公式: https://github.com/ouster-lidar/ouster-ros）
- `~/src/autoware_core` 配下（既存）から `~/ros/jazzy/src/autoware_core` にシンボリックリンク or 直接 clone

---

## マイルストーンとタスクの対応

| Milestone | Tasks |
|---|---|
| M-1 環境準備 | Task 1 |
| M0 事前作業 | Task 2-5 |
| M1 センサー統合 | Task 6-13 |
| M2 Localization | Task 14-19 |
| M3 Vehicle Interface | Task 20-37 |
| M4 Planning | Task 38-42 |
| M5 閉ループ + 検証 | Task 43-46 |

---

## Tasks

### Task 1: Workspace bootstrap（M-1）

**目的:** Thor 上の `~/ros/jazzy` ワークスペースに必要なリポジトリを揃え、apt 依存を入れ、autoware_core / kachaka-api / autoware_rviz_plugins / ouster-ros を一通りビルドできる状態にする。これは「コードを書く」フェーズではなく「環境を整える」フェーズ。

**Files:**
- 修正なし（外部リポジトリの clone と apt install のみ）

- [ ] **Step 1: 必要な apt 依存を入れる（仕様書 jazzy_build_caveats.md と同じ）**

実行コマンド:
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

期待結果: 全パッケージが `Setting up ...` で完了、エラーなし

- [ ] **Step 2: `~/ros/jazzy/src` を作って autoware_core / kachaka-api をリンク**

実行コマンド:
```bash
mkdir -p ~/ros/jazzy/src
cd ~/ros/jazzy/src
ln -sf ~/src/autoware_core autoware_core
ln -sf ~/src/kachaka-api kachaka-api
```

期待結果: `ls -la ~/ros/jazzy/src/` で 2 つのシンボリックリンクが見える

- [ ] **Step 3: `autoware_rviz_plugins` を clone**

実行コマンド:
```bash
cd ~/ros/jazzy/src
git clone https://github.com/autowarefoundation/autoware_rviz_plugins.git
```

期待結果: `ls ~/ros/jazzy/src/autoware_rviz_plugins/package.xml` で見える

- [ ] **Step 4: `ouster-ros` を clone（jazzy ブランチ）**

実行コマンド:
```bash
cd ~/ros/jazzy/src
git clone -b ros2 https://github.com/ouster-lidar/ouster-ros.git
git -C ouster-ros submodule update --init --recursive
```

期待結果: `ls ~/ros/jazzy/src/ouster-ros/ouster_ros/package.xml` で見える

- [ ] **Step 5: rosdep でパッケージ依存を解決**

実行コマンド:
```bash
cd ~/ros/jazzy
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -y --rosdistro=jazzy
```

期待結果: `All required rosdeps installed successfully`

- [ ] **Step 6: kachaka_grpc_ros2_bridge 用の gen-src を生成**

実行コマンド:
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

期待結果: `kachaka-api.grpc.pb.cc kachaka-api.grpc.pb.h kachaka-api.pb.cc kachaka-api.pb.h` の 4 ファイル

- [ ] **Step 7: ベースラインビルド（kachaka_interfaces / kachaka_description / kachaka_grpc_ros2_bridge / autoware_core 系の最低限）**

実行コマンド:
```bash
cd ~/ros/jazzy
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to \
  kachaka_grpc_ros2_bridge \
  autoware_core \
  autoware_rviz_plugins
```

期待結果: `Summary: ... packages finished` でエラー 0、警告は許容

- [ ] **Step 8: 動作確認 — kachaka_grpc_ros2_bridge が起動する**

実行コマンド:
```bash
cd ~/ros/jazzy
source install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka &
sleep 5
ros2 topic list | grep -E "(kachaka|tf)"
kill %1 2>/dev/null
```

期待結果: `/kachaka/odometry/odometry`, `/kachaka/imu/imu`, `/tf` などのトピックが出る

- [ ] **Step 9: コミット（修正がある場合のみ）**

修正がなければ skip。本タスクは外部依存セットアップなのでリポジトリへの変更は基本的に発生しない。

---

### Task 2: M0-A — pointcloud_map の作成

**目的:** OS-1 128 単独で自宅をマッピングし、`pointcloud_map.pcd` + `pointcloud_map/metadata.yaml` を生成する。Kachaka の 2D LiDAR は故障しているため使えない。

**Files:**
- 出力: `~/maps/kachaka_home/pointcloud_map.pcd`
- 出力: `~/maps/kachaka_home/pointcloud_map/metadata.yaml`

- [ ] **Step 1: SLAM ツールを 1 つ選んで `~/ros/jazzy/src` に clone**

候補（仕様書 §3.3 / 16）:
- `glim`（Jazzy対応・3D LiDAR + IMU、推奨）: https://github.com/koide3/glim
- `fast_lio`: https://github.com/hku-mars/FAST_LIO
- `lio_sam`: https://github.com/TixiaoShan/LIO-SAM

`glim` を採用するなら:
```bash
cd ~/ros/jazzy/src
git clone https://github.com/koide3/glim.git
git clone https://github.com/koide3/glim_ros2.git
```

期待結果: clone 完了。READMEの apt 依存を読んで揃える。

- [ ] **Step 2: SLAM ツールをビルド**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --symlink-install --packages-up-to glim_ros2
```

期待結果: ビルド成功

- [ ] **Step 3: OS-1 128 を Thor に有線接続し、ouster-ros driver で点群を発行**

実行コマンド:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch ouster_ros sensor.launch.xml \
  sensor_hostname:=os-XXXXXX.local \
  metadata:=/tmp/os1_meta.json &
sleep 5
ros2 topic hz /ouster/points
```

期待結果: 約 10-20 Hz で点群がpublishされる。`/ouster/imu` も同様に出る。

OS-1 のホスト名 / IP はユーザー固有なので `os-XXXXXX.local` を実機の値に置き換える。

- [ ] **Step 4: Kachaka に OS-1 を載せて手押し or テレオペでマッピング走行を行う**

走行手順（手動操作、コードなし）:
1. Kachaka を `set_manual_control_enabled(true)` にしてから手押し or `kachaka_grpc_ros2_bridge/manual_control` 経由でテレオペ
2. 自宅の通行可能領域を全部回る（直線・曲がり・部屋の隅まで）
3. 走行中に SLAM ツールで pointcloud_map を生成（glim なら `glim_rosnode` を起動した状態で rosbag record も並行）
4. ループクローズが取れる経路で開始点に戻る

期待結果: rosbag に OS-1 点群と IMU が完全に録画される。

- [ ] **Step 5: pointcloud_map.pcd を保存し metadata.yaml を作成**

glim の場合:
```bash
mkdir -p ~/maps/kachaka_home/pointcloud_map
# glim の出力を ~/maps/kachaka_home/pointcloud_map.pcd に保存
# metadata.yaml は autoware_map_loader が要求する形式で手書き
cat > ~/maps/kachaka_home/pointcloud_map/metadata.yaml <<'EOF'
x_resolution: 50.0
y_resolution: 50.0
A.pcd: [0, 0]
EOF
```

期待結果: `~/maps/kachaka_home/` 配下に `pointcloud_map.pcd` と `pointcloud_map/metadata.yaml` がある。pointcloud_map.pcd はファイルサイズ 100 MB 〜 数 GB（屋内範囲依存）。

実装ノート: Autoware の `autoware_map_loader` は分割マップを期待するので、上記 metadata.yaml は **単一マップを 1 タイル** として扱う最小設定。詳細は `autoware_map_loader` のドキュメント参照。

- [ ] **Step 6: コミットなし — マップは外部に置く（リポジトリに含めない）**

理由: pointcloud_map.pcd はリポジトリには大きすぎる。`kachaka_autoware_maps/README.md` に置き場所を書く（Task 5 で作成）。

---

### Task 3: M0-B — lanelet2 vector_map の作成

**目的:** Vector Map Builder で自宅の通行可能領域に最小限のレーンを引き、`lanelet2_map.osm` + `map_projector_info.yaml` を生成する。pointcloud_map と同一の local projection 原点で生成する。

**Files:**
- 出力: `~/maps/kachaka_home/lanelet2_map.osm`
- 出力: `~/maps/kachaka_home/map_projector_info.yaml`

- [ ] **Step 1: Vector Map Builder（TIER IV、Web ツール）を開く**

ブラウザで https://tools.tier4.jp/vector_map_builder_ll2/ を開く（または最新URLは TIER IV ドキュメント参照）。

- [ ] **Step 2: pointcloud_map.pcd を import して背景表示**

Vector Map Builder の「Load PCD」で Task 2 で作った `pointcloud_map.pcd` を読み込む。

期待結果: 自宅の点群が画面上に表示される。

- [ ] **Step 3: 通行可能領域に Lane を最小限引く**

操作:
1. 1 部屋から別部屋への直線レーンを 1 〜 2 本引く（最小限）
2. 各 Lane の幅は Kachaka の幅（0.387m）+ マージンで `0.6m` 程度
3. 速度制限は `0.3 m/s`（Kachaka の最大線速度）

期待結果: lanelet2 上で route が引ける状態。

- [ ] **Step 4: lanelet2_map.osm を export**

Vector Map Builder の「Export」で `lanelet2_map.osm` をダウンロードし、`~/maps/kachaka_home/lanelet2_map.osm` に保存。

期待結果: ファイルが保存される（通常 KB オーダー）。

- [ ] **Step 5: map_projector_info.yaml を作成**

実行コマンド:
```bash
cat > ~/maps/kachaka_home/map_projector_info.yaml <<'EOF'
projector_type: Local
vertical_datum: WGS84
EOF
```

注意: `projector_type: Local` は GNSS 不要の屋内向け設定。Vector Map Builder の export 時に「Local Cartesian」を選んだ前提。

期待結果: ファイルが保存される。

- [ ] **Step 6: pointcloud_map との原点整合確認**

確認手順（人手）:
1. RViz で pointcloud_map.pcd（pcl_ros の `pcd_to_pointcloud`）と lanelet2_map（`autoware_lanelet2_map_visualizer`）を同じ座標系（map）で重ねて表示
2. 部屋の壁の位置と lane の位置がずれていないか目視確認
3. ずれていたら Vector Map Builder で再調整して export しなおす

期待結果: pointcloud_map の壁面と lanelet2 のレーンが整合している。

- [ ] **Step 7: コミットなし**

マップはリポジトリには含めない。

---

### Task 4: M0-C — OS-1 物理固定とキャリブ値取得

**目的:** Ouster OS-1 をシェルフ天面に物理的に固定し、`shelf_top → os1_sensor` のオフセット値とシェルフ自身の概略寸法を実測する。Task 6 の `_shelf_3tier.urdf.xacro` の param と Task 7 の `_ouster_os1.urdf.xacro` の `<origin>` に転記する。

**Files:**
- 出力: メモ（Task 6 のシェルフ寸法 default、Task 7 の OS-1 取付 origin に転記する数値）

- [ ] **Step 1: OS-1 をシェルフ天面の中央寄りに固定**

物理作業（コードなし）。固定方法はユーザー裁量。注意点:
- センサーの「前向き」マークが Kachaka の前進方向と一致するように
- 水平を保つ（傾くと NDT のマッチング精度が落ちる）
- シェルフ天面の中心からの xy ずれを最小化（モデル化ずれを減らす）

- [ ] **Step 2: メジャーで実寸を測る**

人手で計測（後で URDF に反映する数値）:
- **シェルフ自体の寸法**:
  - depth (x 方向、前後): メジャー実測。default 0.32 m に対して合っているか
  - width (y 方向、左右): default 0.38 m に対して合っているか
  - height (z 方向、シェルフ底面〜天板上面): default 0.50 m に対して合っているか
- **OS-1 のシェルフ天面に対する取付オフセット**（`shelf_top` 基準）:
  - x: シェルフ天面中心から OS-1 取付中心までの前方距離（中央なら 0）
  - y: 同 横方向距離
  - z: シェルフ天面（板の上面）から OS-1 sensor 原点（円筒下面）までの高さ。通常 0（直接乗せる）
  - roll/pitch/yaw: 通常 0
- **base_footprint → docking_link → shelf_base** はすべて (0,0,0) で固定（kachaka_description / _shelf_3tier の前提）。base_footprint から見た OS-1 lidar の高さは「docking_link 高さ + shelf height + OS-1 body_height + lidar_to_sensor_z」で URDF が自動算出する

期待結果: 数値メモ（例 `shelf depth=0.32 / width=0.38 / height=0.50, os1_offset_xyz=0,0,0, rpy=0,0,0`）

- [ ] **Step 3: 計測精度を確認**

最初は ±2cm / ±2deg 精度で十分。NDT が収束しない場合に M2 で再調整する。

期待結果: メモ確定。

- [ ] **Step 4: コミットなし**

物理作業のみ。Task 6 のシェルフ寸法 default と Task 7 の OS-1 origin に値を反映する。

---

### Task 5: kachaka_autoware_maps パッケージ作成 & マップ手順書

**目的:** マップは外部 (`~/maps/kachaka_home/`) に置く運用なので、リポジトリには手順書だけ置く `kachaka_autoware_maps` パッケージを作る。

**Files:**
- Create: `ros2/kachaka_autoware_maps/package.xml`
- Create: `ros2/kachaka_autoware_maps/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_maps/README.md`

- [ ] **Step 1: package.xml を作成**

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

- [ ] **Step 2: CMakeLists.txt を作成**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_maps/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.5)
project(kachaka_autoware_maps)

find_package(ament_cmake REQUIRED)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
```

- [ ] **Step 3: README.md を書く**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_maps/README.md`:

```markdown
# kachaka_autoware_maps

Kachaka を Autoware Core で動かすための地図（pointcloud_map.pcd と lanelet2_map.osm）の作成手順書。マップ本体はリポジトリに含まず、ユーザー環境の `~/maps/<location_name>/` 配下に置く運用。

## 前提

- OS-1 128（または同等の 3D LiDAR）が Kachaka に固定されている
- ouster-ros driver で点群が発行できる
- **Kachaka の 2D LiDAR は使用しない**（自宅機は故障している前提）

## ディレクトリ構成

```
~/maps/<location_name>/
├── pointcloud_map.pcd
├── pointcloud_map/
│   └── metadata.yaml
├── lanelet2_map.osm
└── map_projector_info.yaml
```

`autoware_core_map.launch.xml` の `lanelet2_map_path`, `pointcloud_map_path`, `pointcloud_map_metadata_path`, `map_projector_info_path` 引数にこれらを渡す。

## 1. pointcloud_map.pcd の作成

OS-1 単独で 3D SLAM を実行（推奨ツール: `glim`）。詳細は `docs/superpowers/specs/2026-05-02-kachaka-autoware-core-design.md` の §3.3 を参照。

## 2. lanelet2_map.osm の作成

[TIER IV Vector Map Builder](https://tools.tier4.jp/vector_map_builder_ll2/) を使い、pointcloud_map.pcd を背景に lane を引く。**pointcloud_map と同一の local projection 原点で export する**こと。

## 3. map_projector_info.yaml

```yaml
projector_type: Local
vertical_datum: WGS84
```

## 4. pointcloud_map/metadata.yaml（単一タイル運用）

```yaml
x_resolution: 50.0
y_resolution: 50.0
A.pcd: [0, 0]
```

実装に即しては `autoware_map_loader` のドキュメントを参照。
```

- [ ] **Step 4: ビルドが通ることを確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_maps
```

期待結果: `Summary: 1 package finished`

- [ ] **Step 5: コミット**

実行コマンド:
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

期待結果: コミット成功

---

### Task 6: kachaka_description の改良 — 純正 3 段シェルフのマクロ追加

**目的:** Kachaka の純正 3 段シェルフは Kachaka の装備品なので `kachaka_description` パッケージに `_shelf_3tier.urdf.xacro` として追加する。既存の `_kachaka.urdf.xacro` / `_values.urdf.xacro` / `kachaka.urdf.xacro` は破壊変更を避ける（既存ユーザーの URDF 出力を変えない）。シェルフは `docking_link` を起点に取り付けるマクロにし、ドッキング・リフトに追従させる。

**Files:**
- Modify: `ros2/kachaka_description/urdf/_materials.urdf.xacro` (シェルフ用マテリアル追加)
- Create: `ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro`

- [ ] **Step 1: マテリアル追加**

`/home/youtalk/src/kachaka-api/ros2/kachaka_description/urdf/_materials.urdf.xacro` の `</robot>` 直前に以下の 2 マテリアルを append:

```xml
  <material name="shelf_board">
    <color rgba="0.85 0.78 0.65 1.0" />
  </material>
  <material name="shelf_post">
    <color rgba="0.15 0.15 0.15 1.0" />
  </material>
```

最終形:

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

- [ ] **Step 2: `_shelf_3tier.urdf.xacro` を作成**

`/home/youtalk/src/kachaka-api/ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="shelf_3tier" xmlns:xacro="http://ros.org/wiki/xacro">
  <!--
    Kachaka 純正 3 段シェルフのマクロ。
    寸法は概略値。後段で実機計測値に合わせて param 化された値を渡せる。
    `parent` は通常 docking_link。docking lift に追従するため。
  -->
  <xacro:macro name="shelf_3tier"
               params="parent
                       shelf_name:=shelf
                       depth:=0.32
                       width:=0.38
                       height:=0.50
                       board_thickness:=0.015
                       post_size:=0.020">

    <!-- bottom-face center of the shelf is co-located with parent -->
    <link name="${shelf_name}_base_link"/>
    <joint name="${shelf_name}_base_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${shelf_name}_base_link"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
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

- [ ] **Step 3: 既存 `kachaka.urdf.xacro` の出力が変わらないことを確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_description
source install/setup.bash
xacro src/kachaka-api/ros2/kachaka_description/robot/kachaka.urdf.xacro > /tmp/kachaka_after.urdf
grep -c "<link" /tmp/kachaka_after.urdf
grep -c "<joint" /tmp/kachaka_after.urdf
grep "shelf" /tmp/kachaka_after.urdf || echo "no shelf in default kachaka — OK"
```

期待結果: 既存 link/joint 数が変わっていない。`shelf` が出力に含まれない（kachaka.urdf.xacro はシェルフを include しないので）。

- [ ] **Step 4: コミット**

実行コマンド:
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

### Task 7: kachaka_autoware_description パッケージ + Ouster + 統合 URDF

**目的:** 新規パッケージを作り、Ouster OS-1 マクロと、Kachaka + 3 段シェルフ + OS-1 を統合した完全 URDF を作る。

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

- [ ] **Step 3: `_ouster_os1.urdf.xacro` を作成**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/urdf/_ouster_os1.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="ouster_os1" xmlns:xacro="http://ros.org/wiki/xacro">
  <!--
    Ouster OS-1 (128) 簡易モデル。公式の ouster_description が手に入る場合はそれに置き換え可能。
    body: 円筒（直径 85mm、高さ 73.5mm）
    os1_lidar / os1_imu フレームのオフセットは Ouster ICD の値。
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

注意: `_ouster_os1.urdf.xacro` は `_materials.urdf.xacro` の `black` マテリアルを参照するので、include 順は kachaka 側 → ouster 側 にする必要がある（Step 4 で対応）。

- [ ] **Step 4: `kachaka_with_shelf.urdf.xacro` を作成**

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
    3-tier shelf attached to docking_link.
    docking_link is a prismatic joint (lift mechanism); placing the shelf as its
    child means the shelf rises and falls with the lift, matching real operation.
  -->
  <xacro:shelf_3tier parent="docking_link"/>

  <!-- Ouster OS-1 mounted at the shelf top -->
  <xacro:ouster_os1 parent="shelf_top">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:ouster_os1>

</robot>
```

- [ ] **Step 5: ビルドと xacro 展開検証**

実行コマンド:
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

期待結果:
- `xacro exit: 0`
- link 数 ≥ 18（Kachaka 既存 9 link + shelf 9 link + os1 3 link）
- joint 数 ≥ 17
- `shelf_*`, `os1_*` の link がそれぞれ少なくとも 1 件ずつ grep でヒットする

- [ ] **Step 6: コミット**

実行コマンド:
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

### Task 8: vehicle_info.param.yaml と robot_description.launch.py

**目的:** `simple_pure_pursuit` と planning が読む `vehicle_info` を Kachaka 差動駆動向けに作成。`robot_state_publisher` を起動する launch も用意。

**Files:**
- Create: `ros2/kachaka_autoware_description/config/vehicle_info.param.yaml`
- Create: `ros2/kachaka_autoware_description/launch/robot_description.launch.py`

- [ ] **Step 1: vehicle_info.param.yaml を作成（仕様書 §9.3 の値を転記）**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_description/config/vehicle_info.param.yaml`:

```yaml
/**:
  ros__parameters:
    wheel_radius: 0.045
    wheel_width: 0.025
    wheel_base: 0.30
    wheel_tread: 0.20
    front_overhang: 0.15
    rear_overhang: 0.10
    left_overhang: 0.05
    right_overhang: 0.05
    vehicle_height: 1.20
    max_steer_angle: 1.5708
```

- [ ] **Step 2: robot_description.launch.py を作成**

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

- [ ] **Step 3: ビルド確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_description
```

期待結果: ビルド成功

- [ ] **Step 4: launch 単独動作確認**

実行コマンド:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_description robot_description.launch.py &
sleep 3
ros2 topic echo --once /robot_description | head -3
ros2 run tf2_tools view_frames -o /tmp/frames &
sleep 5
kill %1 %2 2>/dev/null
```

期待結果: `/robot_description` トピックに URDF が流れる。TF tree に base_footprint → base_link → docking_link → shelf_base_link → shelf_top → os1_sensor が見える。`ros2 run tf2_ros tf2_echo base_footprint os1_sensor` で transform が取れる（z は 約 0.107 (docking_link) + 0.50 (shelf height) ≈ 0.61 m）

- [ ] **Step 5: コミット**

実行コマンド:
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

### Task 9: ouster-ros launch ラッパー

**目的:** OS-1 128 を `/sensing/lidar/top/pointcloud_raw_ex` というAutoware が期待するトピック名で発行する launch を作る。

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/sensor_ouster.launch.xml`
- ほかは Task 10 で `kachaka_autoware_bridge` パッケージ全体を作る

このタスクは Task 10 のパッケージ作成と一緒にやるため、**Task 10 にマージする**。

---

### Task 10: kachaka_autoware_bridge パッケージのスケルトン + sensor launch

**目的:** メタパッケージ `kachaka_autoware_bridge` を作成し、最初の launch（OS-1 wrapper）を入れる。

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

注意: `ouster-ros` の launch ファイル名と arg 名は本家の最新 README で確認すること。`sensor.launch.xml` は ros2 ブランチの命名。

- [ ] **Step 4: ビルド確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

期待結果: ビルド成功

- [ ] **Step 5: launch 単独動作確認（OS-1 が接続されている場合）**

実行コマンド:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge sensor_ouster.launch.xml \
  sensor_hostname:=<実機ホスト名> &
sleep 8
ros2 topic hz /sensing/lidar/top/pointcloud_raw_ex
kill %1
```

期待結果: 約 10-20 Hz で点群が出る。

- [ ] **Step 6: コミット**

実行コマンド:
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

### Task 11: TF / 点群統合の実機検証（M1 完了条件）

**目的:** robot_state_publisher と ouster driver を起動して、RViz2 で base_footprint 基準で点群が見えることを確認。

- [ ] **Step 1: 統合起動（手動でターミナル 3 つ）**

ターミナル 1（Kachaka bridge）:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka
```

ターミナル 2（OS-1）:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge sensor_ouster.launch.xml sensor_hostname:=<実機>
```

ターミナル 3（URDF）:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_description robot_description.launch.py
```

- [ ] **Step 2: TF tree を確認**

実行コマンド:
```bash
ros2 run tf2_ros tf2_echo base_footprint os1_sensor
ros2 run tf2_ros tf2_echo base_footprint shelf_top
ros2 run tf2_ros tf2_echo shelf_top os1_sensor
```

期待結果: `base_footprint → os1_sensor` の z は概ね 0.50 m（shelf height）+ 0 (os1 mount offset) ≈ 0.50 m。`shelf_top → os1_sensor` は Task 4 で実測した OS-1 取付オフセット（中央取付なら全成分 0）。

- [ ] **Step 3: RViz2 で点群を可視化**

実行コマンド:
```bash
rviz2 &
```

RViz の操作:
- Fixed Frame を `base_footprint` に
- Add → PointCloud2 → Topic `/sensing/lidar/top/pointcloud_raw_ex`
- Add → RobotModel → Description Topic `/robot_description`

期待結果: Kachaka の URDF と OS-1 点群（部屋の点群）が同じ座標系で表示される。

- [ ] **Step 4: コミット — 検証ノート追加（任意）**

このタスクで修正があるとすれば URDF のキャリブ値の微調整:
- シェルフ寸法 default: `kachaka_description/urdf/_shelf_3tier.urdf.xacro` の `depth` / `width` / `height` default を実測値に合わせる
- OS-1 取付 origin: `kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro` の `<xacro:ouster_os1>` の `<origin>` を実測値に合わせる

微調整したらコミット:

```bash
cd ~/src/kachaka-api
git add ros2/kachaka_description/urdf/_shelf_3tier.urdf.xacro \
        ros2/kachaka_autoware_description/urdf/kachaka_with_shelf.urdf.xacro
git commit -m "fix(description): tune shelf and OS-1 calibration values

Adjusted from Task 11 visual inspection in RViz against the live OS-1
point cloud.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

修正なしなら skip。

---

### Task 12: kachaka_autoware_vehicle_interface パッケージのスケルトン

**目的:** TDD でロジックを書く前に、ビルドが通る空のパッケージを作る。

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

- [ ] **Step 2: CMakeLists.txt（最小、コードはまだ無い）**

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

- [ ] **Step 3: 空ディレクトリ + .gitkeep**

実行コマンド:
```bash
cd /home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface
mkdir -p include/kachaka_autoware_vehicle_interface src test launch config
touch include/kachaka_autoware_vehicle_interface/.gitkeep src/.gitkeep test/.gitkeep launch/.gitkeep config/.gitkeep
```

- [ ] **Step 4: ビルド確認（空でも通る）**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

期待結果: `Summary: 1 package finished`

- [ ] **Step 5: コミット**

実行コマンド:
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

### Task 13: M1 完了確認（修正なしのコミットチェックポイント）

**目的:** M1 の完了条件「OS-1 が ROS 2 で発行、TF ツリー完成、RViz で base_footprint 基準の点群が見える」が満たされたことを確認。

- [ ] **Step 1: チェックリスト確認**

確認項目（人手）:
1. `ros2 topic hz /sensing/lidar/top/pointcloud_raw_ex` が 10-20 Hz
2. `ros2 run tf2_ros tf2_echo base_footprint os1_sensor` で transform が取れる
3. RViz2 で `base_footprint` 基準で点群と Kachaka URDF が一致して見える

- [ ] **Step 2: コミットなし**

検証のみ。次は M2 へ。

---

### Task 14: autoware_core_localization の起動 launch ラッパー

**目的:** `autoware_core_localization` を Kachaka の入力（OS-1 + Kachaka wheel_odometry）に合わせて起動する launch を作る。

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/localization.launch.xml`
- Create: `ros2/kachaka_autoware_bridge/config/pose_initializer.param.yaml`

- [ ] **Step 1: pose_initializer.param.yaml を作成（GNSS 無効化版）**

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

- [ ] **Step 2: localization.launch.xml を作成**

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

- [ ] **Step 3: ビルド確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

期待結果: ビルド成功

- [ ] **Step 4: コミット（次の Task 15-19 で動作確認）**

実行コマンド:
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

### Task 15: 静的 vehicle_velocity_converter 統合のための仮 VelocityReport publisher

**目的:** Vehicle Interface ノード（Task 20-）で `/vehicle/status/velocity_status` を発行するが、それ以前に Localization 単独で確認するために、Kachaka の `wheel_odometry` を直接 `VelocityReport` に変換する **暫定relay** を一時的に作る。Task 20 以降で Vehicle Interface ノードに統合する。

実装方針: 暫定 `python` スクリプトで `wheel_odometry` を購読、`VelocityReport` を publish する。本実装は Task 25-27 で C++ で書く。

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml`（M2 中の一時的なもの、M3 完了時に削除）

- [ ] **Step 1: launch を `topic_tools` の transform で記述（python ノード作成は避ける）**

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

注意: `topic_tools transform` は ROS 2 Jazzy では launch から扱える。`expression` は Python 評価式。

- [ ] **Step 2: ビルド確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

期待結果: ビルド成功

- [ ] **Step 3: コミット**

実行コマンド:
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

### Task 16: Localization 実機起動と NDT monte carlo 確認

**目的:** Kachaka 静止状態で NDT が初期姿勢を確定し、`/localization/kinematic_state` が出ることを確認。

- [ ] **Step 1: 4 ターミナルで統合起動**

ターミナル 1: Kachaka bridge
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka
```

ターミナル 2: OS-1
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge sensor_ouster.launch.xml sensor_hostname:=<実機>
```

ターミナル 3: URDF + 暫定 velocity relay + localization
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_description robot_description.launch.py &
ros2 launch kachaka_autoware_bridge temp_velocity_relay.launch.xml &
ros2 launch autoware_core_sensing autoware_core_sensing.launch.xml &
ros2 launch kachaka_autoware_bridge localization.launch.xml \
  map_path:=$HOME/maps/kachaka_home
```

ターミナル 4: RViz（開発PC側でも可）
```bash
rviz2
```

- [ ] **Step 2: 初期姿勢を `2D Pose Estimate` で粗く与える（M0 lanelet2 で覚えた起点付近）**

RViz で `2D Pose Estimate` ボタンをクリックし、自宅の地図上で「Kachakaが今いる位置」を矢印で指定。

期待結果: NDT が monte carlo で収束し、`/localization/kinematic_state` が約 50 Hz で出る。

- [ ] **Step 3: 出力を確認**

実行コマンド:
```bash
ros2 topic hz /localization/kinematic_state
ros2 topic echo --once /localization/kinematic_state
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo map base_footprint
```

期待結果:
- 約 50 Hz
- `pose.position` が現実的な値
- `map → odom → base_footprint` の TF が一貫している

- [ ] **Step 4: Kachaka を 1 m 程度手押しして、NDT が追従することを確認**

人手作業: Kachaka を手で押して 1 m 程度動かす。

確認:
- RViz で点群と URDFが地図上で動く
- `/localization/kinematic_state` の `pose.position` が連続的に変化する

期待結果: 追従して動く。divergence 無し。

- [ ] **Step 5: wheel_odometry の妥当性確認**

実行コマンド:
```bash
ros2 topic echo --once /kachaka/wheel_odometry/wheel_odometry
ros2 topic echo --once /vehicle/status/velocity_status
```

確認: Kachaka を 0.1 m/s で手押しすると `longitudinal_velocity` が 0.1 付近を示すか。

期待結果: 妥当な値が出る（仕様書 §6.1 の要検証ポイント）。NG なら Task 17 で IMU フォールバックを実装。

- [ ] **Step 6: 検証メモを追加（成功時）**

検証結果を `docs/superpowers/specs/2026-05-02-kachaka-autoware-core-design.md` の §16 に追記しても良い（任意）。

修正なしならコミット skip。

---

### Task 17: wheel_odometry NG 時の IMU フォールバック（条件付き）

**目的:** Task 16 Step 5 で `wheel_odometry` が信頼できないと判明した場合のみ実装。仕様書 §6.1 の要検証項目への対応。

**前提:** Task 16 で wheel_odometry が問題なく動いた場合、このタスクは **skip**。

**Files（実施時のみ作成）:**
- Modify: `ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml`（IMU 角速度ベースに置換）

実施時の方針: IMU `/kachaka/imu/imu` の `angular_velocity.z` を `heading_rate` に、wheel_odometry の `linear.x` 成分のみを `longitudinal_velocity` に使う合成 relay に変える。実装は Task 15 と同じ `topic_tools transform` を組み替えるだけ。

- [ ] **Step 1: 必要かどうか判断（人手）**

Task 16 Step 5 で `wheel_odometry` が NG なら以下を実施。

- [ ] **Step 2: IMU フォールバック launch（実施時のみ）**

省略（Task 16 で OK だった場合は不要）。実施するなら topic_tools の `transform` で 2 トピック合成は厳しいので、`kachaka_autoware_vehicle_interface` 内で対応する形にする（Task 25 でフラグ追加）。

- [ ] **Step 3: skip するか実施するかコミット**

実施: 暫定launch更新 + コミット。skip: なにもしない。

---

### Task 18: M2 Localization の検証完了

**目的:** M2 完了条件「NDT + EKF が `/localization/kinematic_state` を出す」を確認。

- [ ] **Step 1: チェックリスト**

1. `/localization/kinematic_state` が 50 Hz で出る
2. Kachaka を 1 m 手押しして追従する
3. `wheel_odometry` が `vehicle_velocity_converter` 経由で EKF に流れている
4. `map → odom` TF が一定（diverge していない）

- [ ] **Step 2: コミットなし**

検証のみ。

---

### Task 19: M2→M3 の hand-off（暫定 relay の維持）

**目的:** M3 で本実装に置き換えるまで、暫定 `temp_velocity_relay.launch.xml` は残す。Task 27 で削除する。

- [ ] **Step 1: メモ**

Task 27 で `temp_velocity_relay.launch.xml` を削除予定。M3 完了時にチェック。

- [ ] **Step 2: コミットなし**

---

### Task 20: control_to_twist_converter のテストファースト（その1: 直進）

**目的:** Control → Twist 変換ロジックの最初のテストを書く。`v=0.2 m/s, δ=0` のとき `linear.x=0.2, angular.z=0` になることを検証。

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp` (失敗用に空)
- Create: `ros2/kachaka_autoware_vehicle_interface/test/test_control_to_twist_converter.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: 空のヘッダを置く（テストがコンパイルエラーで落ちる準備）**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_

namespace kachaka_autoware_vehicle_interface {

// Forward declarations only — implementation comes in Task 21.

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_CONTROL_TO_TWIST_CONVERTER_HPP_
```

- [ ] **Step 2: 失敗するテストを書く**

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

- [ ] **Step 3: CMakeLists.txt にテスト登録**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt` を以下に置き換え:

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

- [ ] **Step 4: テストをビルドして失敗することを確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

期待結果: コンパイルエラー（`ControlToTwistConverter` 未定義）。これが「失敗するテスト」の確認。

- [ ] **Step 5: コミットなし（失敗するテストはコミットしない）**

次の Task 21 で実装してパスさせてから一緒にコミット。

---

### Task 21: control_to_twist_converter の最小実装で Task 20 のテストを通す

**目的:** Task 20 のテストを通す最小実装を書く。

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/control_to_twist_converter.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: ヘッダにクラス定義**

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

- [ ] **Step 2: 実装**

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

- [ ] **Step 3: CMakeLists.txt にライブラリ定義を足す**

`CMakeLists.txt` を更新:

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

- [ ] **Step 4: ビルド & テスト実行**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
colcon test-result --verbose --test-result-base build/kachaka_autoware_vehicle_interface
```

期待結果: `test_control_to_twist_converter` のテスト 1 件 PASS

- [ ] **Step 5: コミット**

実行コマンド:
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

### Task 22: control_to_twist_converter の追加テスト（カーブ・上限飽和・wheel_base=0）

**目的:** 境界値とエッジケースをカバー。

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/test/test_control_to_twist_converter.cpp`

- [ ] **Step 1: 失敗するテストを追加**

`test_control_to_twist_converter.cpp` に以下を append:

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

ファイル先頭の include に `<cmath>` も追加:

```cpp
#include <cmath>
```

- [ ] **Step 2: テスト実行（追加分も全て PASS することを確認）**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
```

期待結果: 5 テスト全部 PASS

- [ ] **Step 3: コミット**

実行コマンド:
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

### Task 23: operation_mode_state_machine のテストファースト

**目的:** STOP / AUTONOMOUS の状態遷移ロジックを TDD で書く。

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/test/test_operation_mode_state_machine.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: 空のヘッダ**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`:

```cpp
#ifndef KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_
#define KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_

namespace kachaka_autoware_vehicle_interface {

// Defined in Task 24.

}  // namespace kachaka_autoware_vehicle_interface

#endif  // KACHAKA_AUTOWARE_VEHICLE_INTERFACE_OPERATION_MODE_STATE_MACHINE_HPP_
```

- [ ] **Step 2: 失敗するテスト（4 ケース）**

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

- [ ] **Step 3: CMakeLists.txt に test を追加**

CMakeLists.txt の `if(BUILD_TESTING)` 内に append:

```cmake
  ament_add_gtest(test_operation_mode_state_machine
    test/test_operation_mode_state_machine.cpp
  )
  target_link_libraries(test_operation_mode_state_machine ${PROJECT_NAME})
```

- [ ] **Step 4: ビルドが失敗することを確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

期待結果: コンパイルエラー（`OperationModeStateMachine` 未定義）

- [ ] **Step 5: コミットなし（次タスクで実装）**

---

### Task 24: operation_mode_state_machine の実装

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/operation_mode_state_machine.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: ヘッダ実装**

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

- [ ] **Step 2: 実装**

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

- [ ] **Step 3: CMakeLists.txt のライブラリ source に追加**

`ament_auto_add_library` のソース行に `src/operation_mode_state_machine.cpp` を append:

```cmake
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/control_to_twist_converter.cpp
  src/operation_mode_state_machine.cpp
)
```

- [ ] **Step 4: テスト実行**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
```

期待結果: 9 テスト全部 PASS（5 + 4）

- [ ] **Step 5: コミット**

実行コマンド:
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

### Task 25: velocity_status_publisher のテストファースト & 実装

**目的:** `nav_msgs/Odometry` → `autoware_vehicle_msgs/VelocityReport` の変換を実装。

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/velocity_status_publisher.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/velocity_status_publisher.cpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/test/test_velocity_status_publisher.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`

- [ ] **Step 1: 失敗するテスト**

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

- [ ] **Step 2: 失敗するヘッダ（最小宣言）**

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

- [ ] **Step 3: 実装**

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

- [ ] **Step 4: CMakeLists.txt に追加**

ライブラリ source に append:
```cmake
  src/velocity_status_publisher.cpp
```

`if(BUILD_TESTING)` 内にテスト追加:
```cmake
  ament_add_gtest(test_velocity_status_publisher
    test/test_velocity_status_publisher.cpp
  )
  target_link_libraries(test_velocity_status_publisher ${PROJECT_NAME})
```

- [ ] **Step 5: ビルド & テスト**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface --event-handlers console_direct+
```

期待結果: 10 テスト PASS

- [ ] **Step 6: コミット**

実行コマンド:
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

### Task 26: vehicle_interface_node の作成（最小起動）

**目的:** rclcpp::Node サブクラスを作り、最低限の起動だけできる状態にする。subscriber/publisher 配線は次タスク以降。

**Files:**
- Create: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`
- Create: `ros2/kachaka_autoware_vehicle_interface/src/main.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/CMakeLists.txt`
- Create: `ros2/kachaka_autoware_vehicle_interface/config/vehicle_interface.param.yaml`
- Create: `ros2/kachaka_autoware_vehicle_interface/launch/vehicle_interface.launch.xml`

- [ ] **Step 1: ヘッダ**

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

- [ ] **Step 2: 実装（最小、パラメータ読み込みのみ）**

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

- [ ] **Step 4: CMakeLists.txt にライブラリ source とexecutableを追加**

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

- [ ] **Step 5: 設定 YAML（仕様書 §9.2）**

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

- [ ] **Step 6: launch ファイル**

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

- [ ] **Step 7: ビルド & 動作確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
source install/setup.bash
ros2 launch kachaka_autoware_vehicle_interface vehicle_interface.launch.xml &
sleep 2
ros2 node info /kachaka_autoware_vehicle_interface
kill %1
```

期待結果: ノードが起動し、`VehicleInterfaceNode started: wheel_base=0.300, vmax=0.300, wmax=1.570` のログが出る。

- [ ] **Step 8: コミット**

実行コマンド:
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

### Task 27: Control サブスクライバ + Twist パブリッシャ + operation_mode ゲート

**目的:** `/control/command/control_cmd` を購読、`/system/operation_mode/state` を見て AUTONOMOUS のときだけ `/kachaka/manual_control/cmd_vel` に流す。

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`

- [ ] **Step 1: ヘッダにメンバ追加**

更新後の `vehicle_interface_node.hpp`:

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

- [ ] **Step 2: 実装更新**

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

注意: 仕様書 §9.1.E では「Vehicle Interface 内に簡易状態機械を実装し、`/system/operation_mode/state` を **発行** する」となっている。本実装では一旦 **購読** のみにし、Task 28 で発行責務を追加する形で段階的に拡張する（autoware_default_adapi の operation_mode 系が core 版にないので、Vehicle Interface が両方やる必要がある）。

- [ ] **Step 3: ビルド & 既存テスト全PASSの確認**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
colcon test --packages-select kachaka_autoware_vehicle_interface
```

期待結果: 既存 10 テスト全 PASS。新規テストはまだ無い。

- [ ] **Step 4: 暫定 temp_velocity_relay の削除**

実行コマンド:
```bash
cd ~/src/kachaka-api
rm ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml
```

理由: M3 で本実装に置き換わるため M2 のhand-offで削除（仕様書 Task 19 のメモ参照）。

ただし、このタスク時点では Vehicle Interface は VelocityReport の発行は未実装なので、削除は **Task 28 の後**にする。Step 4 の削除は **Task 28 完了時に移動**。

→ Step 4 は **skip して Task 28 で実施**。

- [ ] **Step 4 (revised): コミット**

実行コマンド:
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

### Task 28: VelocityReport 発行 + operation_mode 発行 + change_to サービス

**目的:** Vehicle Interface に以下を追加:
- `wheel_odometry` 購読 → `/vehicle/status/velocity_status` 発行（50 Hz）
- `/system/operation_mode/state` 自身が発行（10 Hz）
- `/system/operation_mode/change_to_autonomous`, `change_to_stop` サービス hosting
- `auto_enable_manual_control` で起動時に `set_manual_control_enabled(true)` を呼ぶ

**Files:**
- Modify: `ros2/kachaka_autoware_vehicle_interface/include/kachaka_autoware_vehicle_interface/vehicle_interface_node.hpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/src/vehicle_interface_node.cpp`
- Modify: `ros2/kachaka_autoware_vehicle_interface/package.xml`（`std_srvs` は既にあるので確認）

- [ ] **Step 1: ヘッダ更新**

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

- [ ] **Step 2: 実装更新**

`vehicle_interface_node.cpp` を以下に置き換え:

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

- [ ] **Step 3: ビルド**

実行コマンド:
```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_vehicle_interface
```

期待結果: ビルド成功

- [ ] **Step 4: 単体起動確認**

実行コマンド:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_vehicle_interface vehicle_interface.launch.xml &
sleep 2
ros2 topic list | grep -E "(operation_mode|velocity_status|cmd_vel)"
ros2 service list | grep operation_mode
ros2 topic echo --once /system/operation_mode/state
kill %1
```

期待結果:
- `/system/operation_mode/state`, `/vehicle/status/velocity_status`, `/kachaka/manual_control/cmd_vel` が見える
- `change_to_autonomous`, `change_to_stop` サービスが見える
- `OperationModeState` の `mode: 1`（STOP）が確認できる

- [ ] **Step 5: 暫定 temp_velocity_relay.launch.xml の削除**

実行コマンド:
```bash
cd ~/src/kachaka-api
rm ros2/kachaka_autoware_bridge/launch/temp_velocity_relay.launch.xml
```

- [ ] **Step 6: コミット**

実行コマンド:
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

### Task 29: M3 Vehicle Interface 実機統合確認

**目的:** Vehicle Interface が Kachaka と組み合わせて正しく動くことを確認。

- [ ] **Step 1: 統合起動**

ターミナル A:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_grpc_ros2_bridge grpc_ros2_bridge.launch.xml \
  server_uri:=192.168.1.91:26400 namespace:=kachaka
```

ターミナル B:
```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_vehicle_interface vehicle_interface.launch.xml
```

- [ ] **Step 2: 自動 manual_control 有効化確認**

実行コマンド（別ターミナル）:
```bash
ros2 topic echo --once /vehicle/status/velocity_status
ros2 topic echo --once /system/operation_mode/state
```

期待結果: VelocityReport, OperationModeState が出る。Vehicle Interface 起動ログに `Requested set_manual_control_enabled(true)` がある。

- [ ] **Step 3: AUTONOMOUS 遷移と疑似Control 送信**

実行コマンド:
```bash
# AUTONOMOUS にする
ros2 service call /system/operation_mode/change_to_autonomous \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}"

# 疑似 Control を 1 回送る（v=0.1, delta=0）
ros2 topic pub --once /control/command/control_cmd \
  autoware_control_msgs/msg/Control \
  '{longitudinal: {velocity: 0.1, acceleration: 0.0}, lateral: {steering_tire_angle: 0.0}}'

# 受信できたかどうか
ros2 topic echo --once /kachaka/manual_control/cmd_vel
```

期待結果: `cmd_vel` に `linear.x: 0.1, angular.z: 0.0` が出る。Kachaka 本体が前進する（or `set_manual_control_enabled` が事前に有効化されているなら 0.1 m/s で動く）。**注意: 安全のためまず Kachakaを浮かせておく or 衝突しない場所で実施。**

- [ ] **Step 4: STOP 戻し**

実行コマンド:
```bash
ros2 service call /system/operation_mode/change_to_stop \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}"
ros2 topic pub --once /control/command/control_cmd \
  autoware_control_msgs/msg/Control \
  '{longitudinal: {velocity: 0.1, acceleration: 0.0}, lateral: {steering_tire_angle: 0.0}}'
ros2 topic echo --once /kachaka/manual_control/cmd_vel
```

期待結果: STOP 中は `cmd_vel` が更新されない（または zero Twist のみ）。Kachaka は停止状態。

- [ ] **Step 5: コミット — 検証メモのみ（コード変更なし）**

修正なしなら skip。

---

### Task 30: M3 完了確認

**目的:** M3 の完了条件「Control→Twist 変換、velocity_status、operation_mode 状態機械、ManualControl 自動有効化」を確認。

- [ ] **Step 1: チェックリスト**

1. `colcon test` で全 gtest PASS
2. `vehicle_interface` 起動時 `set_manual_control_enabled(true)` が呼ばれる
3. AUTONOMOUS 時のみ `cmd_vel` が流れる
4. `/vehicle/status/velocity_status` が 50 Hz で出る
5. `/system/operation_mode/state` が 10 Hz で出る
6. `change_to_autonomous`, `change_to_stop` サービスが応答する
7. `cmd_vel_timeout` 経過時に zero Twist が出る

- [ ] **Step 2: コミットなし**

検証のみ。

---

### Task 31: AD-API 起動 launch ラッパー

**目的:** `autoware_core_api.launch.xml` を kachaka 統合用に呼び出す。

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

- [ ] **Step 2: ビルド**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

期待結果: ビルド成功

- [ ] **Step 3: 動作確認（単独）**

```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge api.launch.xml &
sleep 3
ros2 service list | grep -E "/api/(localization|routing|operation_mode)"
kill %1
```

期待結果: `/api/localization/initialize`, `/api/routing/set_route_points` 等が見える（`/api/operation_mode/*` は autoware_core 版には無いので無くて良い）

- [ ] **Step 4: コミット**

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

### Task 32: Planning launch ラッパー

**目的:** `autoware_core_planning.launch.xml` を kachaka_autoware_description の vehicle_info で呼ぶ。

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
    <!-- MVP: ObstacleStop 無効。後段 M6 で有効化 -->
    <arg name="motion_velocity_planner_launch_modules" value="[]"/>
  </include>
</launch>
```

注意: `autoware_core_planning.launch.xml` は `vehicle_model` 引数を要求する場合がある。エラーが出るなら `vehicle_model` を `kachaka` 等のダミー値で渡す（実際には `vehicle_info_param_file` だけ参照される）。

- [ ] **Step 2: ビルド & 単独起動確認**

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

期待結果: ノードが起動するが、map と localization が無いので一部 wait 状態。

- [ ] **Step 3: コミット**

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

### Task 33: Control launch ラッパー

**目的:** `autoware_core_control.launch.xml` を起動。`simple_pure_pursuit` の出力 `/control/command/control_cmd` が Vehicle Interface に届くようにする。

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

- [ ] **Step 2: ビルド & 単独起動**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
source install/setup.bash
ros2 launch kachaka_autoware_bridge control.launch.xml &
sleep 3
ros2 node list | grep simple_pure_pursuit
kill %1
```

期待結果: `simple_pure_pursuit` ノードが起動。

- [ ] **Step 3: コミット**

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

### Task 34: 統合 launch — kachaka_autoware.launch.xml

**目的:** すべてのコンポーネントを 1 launch で起動できるようにする。

**Files:**
- Create: `ros2/kachaka_autoware_bridge/launch/kachaka_autoware.launch.xml`

- [ ] **Step 1: 統合 launch**

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

注意: `vehicle/status/velocity_status` は autoware_core_sensing の `vehicle_velocity_converter` が読む。Vehicle Interface が発行するためロード順は `kachaka_autoware_vehicle_interface` が先（または並行）で良い（DDS は遅延起動を許容）。

- [ ] **Step 2: ビルド**

```bash
cd ~/ros/jazzy
colcon build --packages-select kachaka_autoware_bridge
```

期待結果: ビルド成功

- [ ] **Step 3: コミット（実機検証は次タスク）**

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

### Task 35: M4 Planning 実機検証 + perception_stub の必要性判定

**目的:** Planning が `/planning/trajectory` を出すことを確認。perception 入力が必要なら `perception_stub` を追加する。

- [ ] **Step 1: 統合 launch を起動**

```bash
source ~/ros/jazzy/install/setup.bash
ros2 launch kachaka_autoware_bridge kachaka_autoware.launch.xml \
  server_uri:=192.168.1.91:26400 \
  sensor_hostname:=<実機> \
  map_path:=$HOME/maps/kachaka_home
```

- [ ] **Step 2: 各ノードのログ確認**

別ターミナル:
```bash
ros2 node list | wc -l                           # 多数のノードが見える
ros2 topic list | grep -E "(planning|control|localization)" | head -20
```

確認: `behavior_velocity_planner` や `motion_velocity_planner` が「topic not received」のエラーを出していないか。出していたら perception_stub が必要。

- [ ] **Step 3: 必要に応じて perception_stub 追加**

エラーが `dynamic_objects` (`autoware_perception_msgs/PredictedObjects`)、`occupancy_grid_map`、`traffic_signals` 等の topic 未受信なら、空メッセージを 1 Hz で publish する `perception_stub.launch.xml` を作る:

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
  <!-- 同様に他のトピックも -->
</launch>
```

実際の topic_tools transform は **空入力からの publish ができない**ため、必要なら小さな Python ノードを `kachaka_autoware_bridge/scripts/perception_stub.py` として作る:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from autoware_perception_msgs.msg import PredictedObjects
# 他の必要な型


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

CMakeLists.txt にも以下を追加:
```cmake
install(PROGRAMS scripts/perception_stub.py DESTINATION lib/${PROJECT_NAME})
```

- [ ] **Step 4: 必要なら perception_stub をコミット**

実装した場合のみ:
```bash
cd ~/src/kachaka-api
git add ros2/kachaka_autoware_bridge/scripts/ ros2/kachaka_autoware_bridge/CMakeLists.txt ros2/kachaka_autoware_bridge/launch/perception_stub.launch.xml
git commit -m "feat(bridge): add perception_stub for missing perception inputs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 36: 手動 set_route_points で trajectory 生成確認（M4 完了条件）

**目的:** AD-API の `set_route_points` を CLI から叩いて、`/planning/trajectory` が出ることを確認。

- [ ] **Step 1: 統合 launch 起動済み前提で初期姿勢設定**

```bash
ros2 service call /api/localization/initialize \
  autoware_adapi_v1_msgs/srv/InitializeLocalization \
  "{pose_with_covariance: [{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}, covariance: [0.25, 0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0.0,0,0, 0,0,0,0,0.0,0, 0,0,0,0,0,0.0698]}}]}"
```

期待結果: NDT が monte carlo で初期姿勢を確定。

- [ ] **Step 2: 1 点ゴールを送る**

```bash
ros2 service call /api/routing/set_route_points \
  autoware_adapi_v1_msgs/srv/SetRoutePoints \
  "{header: {frame_id: 'map'}, goal: {position: {x: 1.5, y: 0.0, z: 0.0}, orientation: {w: 1.0}}, waypoints: [], option: {}}"
```

期待結果: trajectory が生成される。

- [ ] **Step 3: trajectory の確認**

```bash
ros2 topic echo --once /planning/trajectory | head -30
ros2 topic hz /planning/trajectory
```

期待結果: trajectory が定期的に publish される。

- [ ] **Step 4: コミットなし（検証）**

---

### Task 37: M5 閉ループ動作（MVP達成）

**目的:** RViz から 1 点指定で Kachaka が目標到達することを確認。

- [ ] **Step 1: rviz 設定ファイル作成**

`/home/youtalk/src/kachaka-api/ros2/kachaka_autoware_bridge/config/autoware.rviz`:

最小設定（autoware_rviz_plugins のパネルが入る）。RViz で対話的に設定して保存するのが楽:

```bash
rviz2
# Display: TF, RobotModel(/robot_description), PointCloud2(/sensing/lidar/top/pointcloud_raw_ex),
#          MarkerArray(/planning/scenario_planning/lane_driving/behavior_planning/path),
#          Path(/planning/trajectory)
# Panels: InitialPoseButtonPanel, RouteTool, EngageButton, AutowareStatePanel
# 保存: File → Save As → autoware.rviz
```

CMakeLists.txt にも install 追加:

```cmake
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
```

- [ ] **Step 2: 統合 launch + RViz 起動**

ターミナル A（Thor上）:
```bash
ros2 launch kachaka_autoware_bridge kachaka_autoware.launch.xml \
  server_uri:=192.168.1.91:26400 \
  sensor_hostname:=<実機> \
  map_path:=$HOME/maps/kachaka_home
```

ターミナル B（開発PC）:
```bash
rviz2 -d $(ros2 pkg prefix kachaka_autoware_bridge)/share/kachaka_autoware_bridge/config/autoware.rviz
```

- [ ] **Step 3: MVP 操作シーケンス（仕様書 §10.3）**

1. RViz の `InitialPoseButtonPanel` で「Initialize」を押す（or `2D Pose Estimate` で粗くポーズ指定）
2. NDT が収束してロボットが地図上の正しい位置に表示される
3. `RouteTool`（or 標準 `2D Goal Pose`）で 1.5 m 先にゴールを置く
4. trajectory が描画される
5. `EngageButton` を押す
6. Kachaka が動き出す → 目標位置に到達 → 自動停止

- [ ] **Step 4: 成功条件確認（仕様書 §13.3）**

- ゴール ± 0.3 m / ± 0.2 rad 以内に到達
- 人手介入なし

5 箇所 × 3 回試行して 80% 以上成功なら MVP 達成。

- [ ] **Step 5: 失敗時の調整候補**

- `wheel_base` 仮想値（vehicle_info.param.yaml）— 旋回が鋭すぎ→大きく、鈍い→小さく
- `simple_pure_pursuit` の `lookahead_gain` / `lookahead_min_distance`
- NDT の voxel_size、iterations
- EKF の Q/R 共分散

調整したら、対応する yaml を編集してコミット:
```bash
git add ros2/kachaka_autoware_description/config/vehicle_info.param.yaml
git commit -m "tune(description): adjust wheel_base virtual value to X.XX based on M5

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 検証完了 → リリースノート / README 追記**

`docs/superpowers/specs/2026-05-02-kachaka-autoware-core-design.md` の §16 リスク欄を「M5 で実機調整完了、wheel_base = X.XX」と書き換えるか、`kachaka_autoware_bridge/README.md` を新規作成して MVP 達成報告を残す。

```bash
git add ros2/kachaka_autoware_bridge/README.md
git commit -m "docs(bridge): MVP M5 closed-loop achieved

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

### Spec coverage

仕様書の各セクションに対応するタスクを確認:

| Spec section | Task |
|---|---|
| §3.3-1 pointcloud_map | Task 2 |
| §3.3-2 lanelet2 | Task 3 |
| §3.3-3 OS-1 物理固定 | Task 4 |
| §4.2 TF tree (docking_link → shelf → os1) | Task 6, 7 (URDF) |
| §4.3 座標系整合 | Task 3 (Vector Map Builder で整合) |
| §5 kachaka_description 改良 | Task 6 |
| §5 新規パッケージ構成 | Task 5, 7, 10, 12 |
| §6 Localization | Task 14, 16 |
| §6.1 wheel_odometry 検証 | Task 16 Step 5, Task 17 |
| §7 Planning | Task 32 |
| §7.1 ゴール受信フロー | Task 31, 36 |
| §8 Control | Task 33 |
| §9.1.A Control→Twist | Task 20-22, 27 |
| §9.1.B cmd_vel ゲート | Task 27 |
| §9.1.C VelocityReport | Task 25, 28 |
| §9.1.D ManualControl 自動 | Task 28 |
| §9.1.E Operation Mode | Task 23-24, 28 |
| §9.2 設定 | Task 26 |
| §9.3 vehicle_info | Task 8 |
| §10 AD-API | Task 31 |
| §10.2 RViz panels | Task 1 (clone), 37 (RViz設定) |
| §10.3 操作シーケンス | Task 37 |
| §11 データフロー | Task 34 (統合 launch) |
| §12 エラー処理 timeout | Task 28 (zero Twist failsafe) |
| §13.1 単体テスト | Task 20-25 |
| §13.3 システムテスト | Task 37 |
| §14 マイルストーン | Task 1-37 全体 |
| §16 リスク wheel_base 調整 | Task 37 Step 5 |
| §16 リスク wheel_odometry | Task 16 Step 5 + Task 17 |

ギャップ確認:
- §13.2 結合テスト（rosbag 回帰）: M5 後の運用フェーズで作る想定で MVP には含めず（仕様書 §2.2 後続フェーズ扱い相当）。OK
- §16 リスク NDT divergence: M5 で実機調整時に対応（Task 37 Step 5）。OK
- §17 オープンな質問: MVP では決定不要、後続でOK
- §16 障害物停止 (M6): MVP外（仕様書 §2.2 OOS）。Task 32 で `motion_velocity_planner_launch_modules: []` で MVP 中は無効化済。OK

### Placeholder scan

- "TBD" / "TODO" / "implement later": なし。実装まちのものは Task 35 の perception_stub だけで、これは実装条件を明記してある（必要なら作る、なら作らない）。
- "Add appropriate error handling": なし（具体的な timeout 値・rate を明記）。
- "Similar to Task N": なし（Task 21-22 は別タスクで内容を完全展開）。
- 不明な型 / 関数: 全て定義タスクが先行する（`ControlToTwistConverter` は Task 21、`OperationModeStateMachine` は Task 24、`convert_odometry_to_velocity_report` は Task 25）。OK

### Type consistency

- `ControlToTwistConverter` / `ControlToTwistParams`: Task 20 ヘッダ宣言 → Task 21 実装。一貫。
- `OperationMode` enum / `OperationModeStateMachine`: Task 23 → 24 で一貫。
- `convert_odometry_to_velocity_report`（snake_case 関数）: Task 25 で一貫。
- `VehicleInterfaceNode`: Task 26 → 27 → 28 で段階的拡張。フィールド名・メソッド名一貫。
- launch 名: `vehicle_interface.launch.xml`, `kachaka_autoware.launch.xml`, `localization.launch.xml`, `planning.launch.xml`, `control.launch.xml`, `api.launch.xml`, `sensor_ouster.launch.xml`. 全て下流から参照される名で一貫。

### スコープチェック

仕様書 M0〜M5 を 37 タスクでカバー。M6 は範囲外（仕様書通り）。タスクは独立して実装でき、各タスクの完了条件が明確。1 つの実装計画で進められる。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-kachaka-autoware-core-mvp.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. M3 のような TDD タスクでは特に有効。

**2. Inline Execution** — このセッション内でバッチ実行、チェックポイントで一時停止。M0 のような実機作業がブロックすると進めない。

**Which approach?**

ただしこの計画には **物理作業 / 実機作業 / 外部ハードウェア依存タスク** が多数含まれます:
- Task 2 (M0 マッピング走行)
- Task 3 (Vector Map Builder Web UI 操作)
- Task 4 (OS-1 物理固定)
- Task 11, 16, 29, 35, 36, 37 (実機検証)

これらはエージェント単独では完了できないため、**コード実装タスク (Task 5-10, 12, 14-15, 20-28, 31-34) と実機タスクを分離**して進めることを推奨します。
