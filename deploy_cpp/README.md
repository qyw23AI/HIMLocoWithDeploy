# deploy_cpp — C++ 真机/仿真部署节点

基于 ROS2 Humble 的四足机器人部署系统，使用 LibTorch 执行 HIM 策略推理，通过 Unitree GO-M8010-6 SDK 控制 12 个关节，并支持 MuJoCo 仿真闭环验证。

本版本已完成以下关键改造：

- 运行时参数统一迁移到 YAML（强制配置文件启动）
- 支持逐关节传动比 joint_transmission_ratio
- 小腿额外 3:7 减速比已纳入控制下发与观测回读
- 支持逐关节 dof_pos_scale、dof_vel_scale、action_scale
- 支持多机器人配置切换（mybot / mybot_v2）
- MuJoCo/URDF 模型路径由 YAML 选择，不再在代码中写死 mybot
- **[新功能]** 手机网页端 UDP 遥控 (Web Teleop)，支持双摇杆速度控制与状态切换

## TODO

- [ ] 重力投影后续改为直接使用 FAST-LIVO2 输出，替代当前仿真中的四元数计算流程，以提升大倾角鲁棒性。

## 目录结构

```text
deploy_cpp/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/
│   ├── robot_config.h              # 编译期常量与历史兼容常量
│   ├── robot_runtime_config.h      # 运行时 YAML 配置结构
│   ├── motor_driver.h
│   ├── imu_subscriber.h
│   ├── policy_runner.h
│   ├── keyboard_controller.h
│   ├── state_machine.h
│   ├── robot_visualizer.h
│   └── Unitree_Motor/
├── src/
│   ├── robot_runtime_config.cpp    # YAML 加载与强校验
│   ├── deploy_node.cpp
│   ├── motor_driver.cpp
│   ├── imu_subscriber.cpp
│   ├── policy_runner.cpp
│   ├── keyboard_controller.cpp
│   ├── state_machine.cpp
│   ├── robot_visualizer.cpp
│   └── motor_debug_node.cpp
├── config/
│   ├── robots/
│   │   ├── mybot.yaml
│   │   └── mybot_v2.yaml
│   ├── mybot.rviz
│   ├── cyclonedds.xml
│   └── setup_cyclonedds.sh
├── launch/
│   ├── deploy.launch.py
│   ├── sim.launch.py
│   ├── visualize.launch.py
│   └── motor_debug.launch.py
├── sim/
│   ├── mujoco_sim_node.py
│   └── requirements.txt
├── robot/
│   ├── mybot/
│   └── mybot_v2/
├── web_teleop/
│   ├── app.py                      # Python Web 中转服务端
│   └── index.html                  # 手机前端 UI (双摇杆)
├── policy/
└── lib/
```

## 依赖

- ROS2 Humble (rclcpp, std_msgs, sensor_msgs)
- LibTorch (PyTorch C++ API)
- Unitree Actuator SDK
- yaml-cpp (C++ 读取 YAML)
- PyYAML (Python 仿真节点读取 YAML)
- robot_state_publisher / rviz2（可视化）

安装示例：

```bash
sudo apt install ros-humble-robot-state-publisher ros-humble-rviz2
sudo apt install libyaml-cpp-dev
```

MuJoCo Python 环境与 Web 遥控服务端：

```bash
conda create -n himloco python=3.10 -y
conda activate himloco
pip install mujoco numpy pyyaml flask flask-socketio eventlet
```

## 配置机制（重点）

### 1. 强制 YAML 启动

deploy_node 现在要求传入 robot_config_file；不再依赖旧的 policy_path/device/port0/port1 启动参数。

必须字段示例（节选）：

```yaml
num_of_dofs: 12
joint_transmission_ratio: [6.33, 6.33, 14.77, 6.33, 6.33, 14.77, 6.33, 6.33, 14.77, 6.33, 6.33, 14.77]
motor_is_reversed: [false, true, true, false, false, false, true, true, true, true, false, false]
policy_path: policy/policy.pt
device: cuda:0
port0: /dev/motor_front
port1: /dev/motor_rear
urdf_relpath: robot/mybot/urdf/mybot.urdf
mujoco_xml_relpath: robot/mybot/xml/mybot.xml
```

### 2. 逐关节传动比含义

统一定义：

- q_motor = q_joint * ratio[i]
- dq_motor = dq_joint * ratio[i]
- tau_motor = tau_joint / ratio[i]
- kp_motor = kp_joint / ratio[i]^2
- kd_motor = kd_joint / ratio[i]^2

其中 ratio[i] 来自 YAML 的 joint_transmission_ratio。

关节方向由 YAML 的 `motor_is_reversed[i]` 控制：

- false: direction = +1
- true: direction = -1

该字段为必填，缺失会在启动时直接报错退出。

当前默认约定：

- 髋/大腿：6.33（保留原电机减速比）
- 小腿：6.33 * 7/3 = 14.77（叠加额外 3:7 减速器后的总传动比）

### 3. 控制链路语义

- 策略输出目标角度是关节侧语义
- 真实电机下发时按 joint_transmission_ratio 做 joint→motor 换算
- SDK 回读编码器数据后按同一 ratio 做 motor→joint 还原
- Policy 观测使用还原后的关节侧 dof_pos/dof_vel

## 多机器人选择（mybot / mybot_v2）

通过切换 robot_config_file 实现。

- mybot 配置：config/robots/mybot.yaml
- mybot_v2 配置：config/robots/mybot_v2.yaml

mybot_v2 默认模型字段：

- urdf_relpath: robot/mybot_v2/urdf/mybot_v2.urdf
- mujoco_xml_relpath: robot/mybot_v2/xml/mybot_v2_for_mujoco.xml
- isaac_xml_relpath: robot/mybot_v2/xml/mybot_v2_for_isaacgym.xml

## 为什么同时有 XML 和 URDF

- MuJoCo 物理仿真只使用 XML（mujoco_xml_relpath）
- URDF 仅用于 robot_state_publisher + RViz 显示 TF/网格

也就是说，URDF 不参与 MuJoCo 动力学计算。

## 构建

```bash
cd ~/humble/Quadruped/HIMLoco
source /opt/ros/humble/setup.bash

colcon build --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch

source install/setup.bash
```
export LD_LIBRARY_PATH=/home/getting/miniconda3/envs/himloco/lib:$LD_LIBRARY_PATH
## 运行

### A. 真机（推荐 launch）

```bash
source /opt/ros/humble/setup.bash
source ~/humble/Quadruped/HIMLoco/install/setup.bash

ros2 launch deploy_cpp deploy.launch.py \
  robot_config_file:=$(ros2 pkg prefix deploy_cpp)/share/deploy_cpp/config/robots/mybot.yaml
```

切换到 mybot_v2：

```bash
ros2 launch deploy_cpp deploy.launch.py \
  robot_config_file:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/config/robots/mybot_v2_real.yaml
```

### B. 真机（直接 run）

```bash
ros2 run deploy_cpp deploy_node --ros-args \
  -p robot_config_file:=$(ros2 pkg prefix deploy_cpp)/share/deploy_cpp/config/robots/mybot.yaml
```

```bash
ros2 run deploy_cpp deploy_node --ros-args -p robot_config_file:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/config/robots/mybot_v2_real.yaml
```


### C. Debug 无电机

```bash
ros2 run deploy_cpp deploy_node --ros-args \
  -p robot_config_file:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/config/robots/mybot_v2_real.yaml \
  -p debug_no_motor:=true
```

### D. MuJoCo 仿真（推荐）


切换 mybot_v2：

```bash
python3 sim/mujoco_sim_node.py \
  --robot-config config/robots/mybot_v2_sim.yaml

ros2 run deploy_cpp deploy_node --ros-args -p robot_config_file:=/home/getting/humble/Quadruped/HIMLoco/deploy_cpp/config/robots/mybot_v2_sim.yaml -p sim_mode:=true -p debug_no_motor:=false
```

### E. 电机调试节点

```bash
ros2 launch deploy_cpp motor_debug.launch.py
```

### F. 手机网页 UDP 遥控 (Web Teleop)

本系统支持通过手机浏览器虚拟双摇杆替代键盘进行控制（前提是 YAML 中已配置 `teleop_udp_enable: true`）。

**步骤 1：启动 C++ 部署节点**
```bash
ros2 run deploy_cpp deploy_node --ros-args -p robot_config_file:=config/robots/mybot_v2_real.yaml
```

**步骤 2：启动 Python Web 中转服务端**
（确保在同一台机器的 `himloco` conda 环境中运行，并放通 5000 端口）
```bash
cd ~/humble/Quadruped/HIMLoco/deploy_cpp/web_teleop
python3 app.py --udp-port 9870 --web-port 5000
```

**步骤 3：手机浏览器访问**
将手机连接至与机载电脑/主控电脑同一局域网（Wi-Fi），在浏览器中输入：
`http://<机载电脑IP>:5000`

- **左侧摇杆**：控制前后平移 (vx) 与左右平移 (vy)
- **右侧摇杆**：控制原地转向 (yaw)
- **顶部按键**：切换 IDLE / STAND UP / RL MODE / DAMPING
- **右上角 STOP**：红色急停按键，立即切入 IDLE 并速度归零

*注：前端操作会自动覆盖键盘控制的速度指令。*

## launch 参数

### deploy.launch.py

- robot_config_file: 机器人 YAML 路径（必需语义）
- debug_no_motor: true/false
- sim_mode: true/false
- sim_pingpong_mode: true/false

### sim.launch.py

- robot_config_file: 机器人 YAML 路径
- sim_pingpong_mode: true/false

sim.launch.py 会从 YAML 中读取 urdf_relpath 并自动启动 robot_state_publisher。

## 键盘控制

- 0: IDLE
- 1: STAND_UP
- 2: RL
- 3: JOINT_DAMPING
- W/S: vx 增减
- Q/E: vy 增减
- A/D: yaw 增减
- R: 速度清零
- Space: 急停回 IDLE
- Esc: 退出

实际步长和速度范围由 YAML 中 cmd_vx_step/cmd_vy_step/cmd_yaw_step 与 cmd_*_min/max 决定。

## QoS

| Topic | Reliability | Depth | Durability |
|---|---|---|---|
| /mujoco/joint_cmd | BEST_EFFORT | 1 | VOLATILE |
| /mujoco/joint_state | BEST_EFFORT | 1 | VOLATILE |
| /fast_livo2/state6 | BEST_EFFORT | 1 | VOLATILE |
| /joint_states | RELIABLE | 10 | VOLATILE |

## 当前关键实现摘要

- 运行时配置入口：robot_runtime_config.cpp
- 逐关节传动比换算：motor_driver.cpp
- 逐关节观测/动作 scale：policy_runner.cpp
- standup 时长/目标改为 YAML：state_machine.cpp
- 键盘速度范围改为 YAML：keyboard_controller.cpp
- 模型路径按 YAML 选择：sim.launch.py + sim/mujoco_sim_node.py

## 已知注意事项

- MuJoCo 只使用 XML；URDF 仅用于 RViz/TF。
- 若使用 ros2 launch 启动 deploy_node 时遇到键盘输入不响应，改用 ros2 run 方式可规避终端 stdin 继承问题。
- 配置缺失关键字段时会在启动阶段直接报错退出（这是设计行为，用于防止 silent fallback）。
