# deploy_cpp — C++ 真机部署节点

基于 ROS2 Humble 的四足机器人 C++ 部署系统，使用 LibTorch 进行 HIM 策略推理，通过 Unitree GO-M8010-6 电机 SDK 控制 12 个关节。

## TODO List

- [] 计算重力投影时，后续采用FAST-LIVO2的输出，替代当前的基于四元数的计算方法，以提高在大倾角下的准确性和鲁棒性。

## 目录结构

```
deploy_cpp/
├── CMakeLists.txt              # 构建配置
├── package.xml                 # ROS2 包描述
├── README.md
├── include/
│   ├── robot_config.h          # 机器人配置常量
│   ├── motor_driver.h          # 电机驱动封装
│   ├── imu_subscriber.h        # ROS2 IMU 订阅器
│   ├── policy_runner.h         # 策略推理 (LibTorch JIT)
│   ├── keyboard_controller.h   # 键盘控制器
│   ├── state_machine.h         # 状态机
│   ├── robot_visualizer.h      # RViz 可视化 (JointState 发布)
│   └── Unitree_Motor/          # Unitree 电机 SDK 头文件
├── src/
│   ├── deploy_node.cpp         # 主入口 + 50Hz 控制循环
│   ├── motor_driver.cpp
│   ├── imu_subscriber.cpp
│   ├── policy_runner.cpp
│   ├── keyboard_controller.cpp
│   ├── state_machine.cpp
│   ├── robot_visualizer.cpp    # RViz JointState 发布实现
│   └── motor_debug_node.cpp    # 电机调试节点 (独立)
├── config/
│   └── mybot.rviz              # RViz2 配置文件
├── launch/
│   ├── deploy.launch.py        # 主部署 launch 文件
│   ├── visualize.launch.py     # RViz 可视化 launch 文件
│   ├── motor_debug.launch.py   # 电机调试 launch 文件
│   └── sim.launch.py           # MuJoCo 仿真模式 launch 文件
├── sim/
│   ├── mujoco_sim_node.py      # MuJoCo 仿真 Python 节点
│   └── requirements.txt        # Python 依赖
├── robot/
│   └── mybot/
│       ├── urdf/mybot.urdf     # 机器人 URDF 描述文件
│       ├── meshes/             # STL/OBJ 模型文件
│       └── xml/mybot.xml       # MuJoCo 模型
├── policy/                     # 放置 policy.pt 文件
└── lib/                        # Unitree 电机 SDK 动态库
```

## 依赖

- **ROS2 Humble** (rclcpp, std_msgs, sensor_msgs)
- **LibTorch** (PyTorch C++ API, ≥ 2.0)
- **Unitree Actuator SDK** (GO-M8010-6 电机通信)
- **robot_state_publisher** (URDF → TF 变换发布)
- **joint_state_publisher** (关节状态发布, 可视化用)
- **rviz2** (3D 可视化)

### 安装 LibTorch

```bash
# 下载 LibTorch (CUDA 版)
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
sudo unzip libtorch-*.zip -d /opt/

# 或 CPU 版
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
sudo unzip libtorch-*.zip -d /opt/
```

### 安装 ROS2 可视化依赖

```bash
sudo apt install ros-humble-robot-state-publisher \
                 ros-humble-joint-state-publisher \
                 ros-humble-joint-state-publisher-gui \
                 ros-humble-rviz2
```

### 安装 MuJoCo 仿真环境（可选）

```bash
conda create -n mujoco_sim python=3.10 -y
conda activate mujoco_sim
pip install mujoco numpy
```

## 构建

```bash
# 进入工作空间
cd ~/humble/Quadruped/HIMLoco

# 设置 ROS2 环境
source /opt/ros/humble/setup.bash

# 构建 (指定 LibTorch 路径)
colcon build --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch

# 加载构建结果
source install/setup.bash
```

## 运行

### 准备 policy.pt

在训练端导出 JIT 模型（在 `play.py` 中设置 `EXPORT_POLICY = True`），将生成的 `policy.pt` 复制到 `deploy_cpp/policy/` 目录。

### 真机运行

```bash
source install/setup.bash

# 方式 1: 直接运行
ros2 run deploy_cpp deploy_node --ros-args \
  -p policy_path:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/policy/policy.pt \
  -p device:=cuda:0 \
  -p port0:=/dev/ttyUSB0 \
  -p port1:=/dev/ttyUSB1

# 方式 2: 使用 launch 文件
ros2 launch deploy_cpp deploy.launch.py \
  policy_path:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/policy/policy.pt \
  device:=cuda:0
```

### Debug 模式（无电机）

```bash
ros2 run deploy_cpp deploy_node --ros-args \
  -p debug_no_motor:=true \
  -p policy_path:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/policy/policy.pt \
  -p device:=cuda:0
```

### RViz 可视化

在 RViz 中实时同步显示机器人关节状态，可与 deploy_node 配合使用。

```bash
# 方式 1: 单独查看默认站立姿态
ros2 launch deploy_cpp visualize.launch.py

# 方式 2: 配合 deploy_node 显示实时状态
#   终端 1 — 启动 RViz（关闭内置关节发布器）
ros2 launch deploy_cpp visualize.launch.py use_jsp:=false
#   终端 2 — 启动控制节点
ros2 launch deploy_cpp deploy.launch.py debug_no_motor:=true
```

`visualize.launch.py` 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_jsp` | `true` | 启动 joint_state_publisher（配合 deploy_node 时设为 false） |
| `use_gui` | `false` | 使用 GUI 滑块手动调节关节角度 |

### MuJoCo 仿真模式

使用 MuJoCo 物理引擎验证完整控制流程（状态机 + 策略推理 + PD 控制），无需实体电机。

C++ deploy_node 通过 ROS2 话题 `/mujoco/joint_cmd` 和 `/mujoco/joint_state` 与 Python MuJoCo 仿真节点通信，控制逻辑与真机完全一致。

```bash
# 终端 1 — 启动 MuJoCo 仿真（Python，需要 conda 环境）
cd ~/humble/Quadruped/HIMLoco/deploy_cpp
conda activate mujoco_sim
source /opt/ros/humble/setup.bash
python3 sim/mujoco_sim_node.py -p pingpong_mode:=true 

# 终端 2 — 启动控制节点（注意：必须用 ros2 run，不能用 launch，否则键盘无响应）
source /opt/ros/humble/setup.bash
source ~/humble/Quadruped/HIMLoco/install/setup.bash
ros2 run deploy_cpp deploy_node --ros-args \
  -p sim_mode:=true \
  -p policy_path:=$(ros2 pkg prefix deploy_cpp)/share/deploy_cpp/policy/policy.pt \
  -p device:=cuda:0
  -p pingpong_mode:=true  

# 终端 3 — robot_state_publisher（可选，用于 RViz）：
source /opt/ros/humble/setup.bash
source ~/humble/Quadruped/HIMLoco/install/setup.bash
ros2 launch deploy_cpp visualize.launch.py use_jsp:=false
```

> **注意**：仿真模式下必须用 `ros2 run` 直接运行 deploy_node，不能用 `ros2 launch`，否则键盘输入无法传递给进程。

### 电机调试

独立节点，不需要 IMU、策略或键盘。发送全零参数 (q=0, dq=0, kp=0, kd=0, tau=0) 给所有电机，仅读取并打印电机角度，同时在 RViz 中实时显示关节状态，用于验证各关节映射是否正确。

```bash
# 默认启动（电机调试 + RViz 可视化）
ros2 launch deploy_cpp motor_debug.launch.py

# 指定串口
ros2 launch deploy_cpp motor_debug.launch.py \
  port0:=/dev/ttyUSB0 \
  port1:=/dev/ttyUSB1

# 仅终端输出，不启动 RViz
ros2 launch deploy_cpp motor_debug.launch.py rviz:=false

# 直接运行（无 RViz，需另行启动 visualize.launch.py）
ros2 run deploy_cpp motor_debug_node --ros-args \
  -p port0:=/dev/ttyUSB0 \
  -p port1:=/dev/ttyUSB1 \
  -p rate_hz:=10.0
```

`motor_debug.launch.py` 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `port0` | `/dev/ttyUSB0` | 前腿串口 |
| `port1` | `/dev/ttyUSB1` | 后腿串口 |
| `rate_hz` | `10.0` | 读取频率 (Hz) |
| `rviz` | `true` | 是否启动 RViz 可视化 |

输出示例：
```
Joint             Angle(rad)  Angle(deg)  Vel(rad/s)
------------------------------------------------------
FR_hip_joint      -0.0982     -5.6273     0.0012
FR_thigh_joint     0.7856     45.0000     0.0000
FR_calf_joint     -1.5012    -86.0048    -0.0003
...
```

## 操作说明

| 按键 | 功能 |
|------|------|
| `0` | 切换到 Idle（零力矩） |
| `1` | 切换到 StandUp（缓慢站起，2秒） |
| `2` | 切换到 RL（策略控制） |
| `3` | 切换到 JointDamping（关节阻尼缓冲） |
| `W/S` | 增减前进速度 vx（±0.1 m/s） |
| `A/D` | 增减偏航角速度 yaw（±0.3 rad/s） |
| `Q/E` | 增减侧移速度 vy（±0.1 m/s） |
| `R` | 重置速度为零 |
| `Space` | 紧急停止 → Idle |
| `Esc` | 退出程序 |

## 典型使用流程

```
启动 → IDLE（趴着）
按 1 → STAND_UP（缓慢站起，2秒）
按 2 → RL（策略运行，WASD 控制行走）
按 3 → JOINT_DAMPING（减速停下）
按 0 → IDLE（趴下）
紧急：任何时候按 Space → IDLE
```

## 状态转移图

```
IDLE ──────→ STAND_UP (按 1)
STAND_UP ──→ RL (按 2，站起完成后)
STAND_UP ──→ JOINT_DAMPING (按 3)
RL ────────→ JOINT_DAMPING (按 3)
RL ────────→ STAND_UP (按 1)
JOINT_DAMPING → IDLE (按 0)
JOINT_DAMPING → STAND_UP (按 1)
任何状态 ──→ IDLE (Space 紧急停止)
```

## 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 控制频率 | 50 Hz (0.02s) | legged_robot_config |
| PD 增益 (关节侧) | Kp=40, Kd=1 | mybot_config |
| PD 增益 (电机侧) | kp≈0.999, kd≈0.025 | Kp/r², r=6.33 |
| Action Scale | 0.25 | mybot_config |
| 观测维度 | 45 × 6 = 270 | legged_robot_config |
| 减速比 | 6.33 | GO-M8010-6 |

## 电机映射

| DOF | 关节 | 电机ID | 串口 | 反向 |
|-----|------|--------|------|------|
| 0 | FR_hip | 1 | USB0 | ❌ |
| 1 | FR_thigh | 2 | USB0 | ✅ |
| 2 | FR_calf | 3 | USB0 | ✅ |
| 3 | FL_hip | 4 | USB0 | ❌ |
| 4 | FL_thigh | 5 | USB0 | ❌ |
| 5 | FL_calf | 6 | USB0 | ❌ |
| 6 | RR_hip | 10 | USB1 | ✅ |
| 7 | RR_thigh | 11 | USB1 | ✅ |
| 8 | RR_calf | 12 | USB1 | ✅ |
| 9 | RL_hip | 7 | USB1 | ✅ |
| 10 | RL_thigh | 8 | USB1 | ❌ |
| 11 | RL_calf | 9 | USB1 | ❌ |
