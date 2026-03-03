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
│   └── state_machine.h         # 状态机
├── src/
│   ├── deploy_node.cpp         # 主入口 + 50Hz 控制循环
│   ├── motor_driver.cpp
│   ├── imu_subscriber.cpp
│   ├── policy_runner.cpp
│   ├── keyboard_controller.cpp
│   └── state_machine.cpp
├── launch/
│   └── deploy.launch.py        # ROS2 launch 文件
└── policy/                     # 放置 policy.pt 文件
```

## 依赖

- **ROS2 Humble** (rclcpp, sensor_msgs)
- **LibTorch** (PyTorch C++ API, ≥ 2.0)
- **Unitree Actuator SDK** (GO-M8010-6 电机通信)

### 安装 LibTorch

```bash
# 下载 LibTorch (CUDA 版)
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
sudo unzip libtorch-*.zip -d /opt/

# 或 CPU 版
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
sudo unzip libtorch-*.zip -d /opt/
```

## 构建

```bash
# 进入工作空间
cd ~/humble/Quadruped/HIMLoco

# 设置 ROS2 环境
source /opt/ros/humble/setup.bash

# 构建 (指定 LibTorch 和 Unitree SDK 路径)
colcon build --packages-select deploy_cpp 

# 或者使用 CMake 直接构建
mkdir -p deploy_cpp/build && cd deploy_cpp/build
cmake .. \
  -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
  -DUNITREE_SDK_DIR=$HOME/Unitree_Motor/unitree_actuator_sdk
make -j$(nproc)
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
