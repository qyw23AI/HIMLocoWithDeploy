cmake_minimum_required(VERSION 3.14)
project(deploy_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# ========================== Dependencies ==========================

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# LibTorch (PyTorch C++ API)
# Set Torch_DIR if not found automatically:
#   cmake -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..
find_package(Torch REQUIRED)

# Unitree Actuator SDK (headers and libs bundled in project)
set(UNITREE_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/Unitree_Motor")
set(UNITREE_SDK_LIB_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/lib")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(UNITREE_SDK_LIB "${UNITREE_SDK_LIB_DIR}/libUnitreeMotorSDK_Arm64.so")
else()
    set(UNITREE_SDK_LIB "${UNITREE_SDK_LIB_DIR}/libUnitreeMotorSDK_Linux64.so")
endif()

if(NOT EXISTS "${UNITREE_SDK_LIB}")
    message(FATAL_ERROR "Unitree SDK library not found: ${UNITREE_SDK_LIB}")
endif()

# ========================== Include Directories ==========================

include_directories(
    include
    ${UNITREE_SDK_INCLUDE_DIR}
)

# ========================== Sources ==========================

set(SOURCES
    src/motor_driver.cpp
    src/imu_subscriber.cpp
    src/policy_runner.cpp
    src/keyboard_controller.cpp
    src/state_machine.cpp
    src/deploy_node.cpp
)

# ========================== RPATH ==========================
# Ensure the installed binary can find .so libs at runtime
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;/opt/libtorch/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# ========================== Executable ==========================

add_executable(deploy_node ${SOURCES})

target_link_libraries(deploy_node
    ${UNITREE_SDK_LIB}
    ${TORCH_LIBRARIES}
)

ament_target_dependencies(deploy_node
    rclcpp
    std_msgs
)

# ========================== Install ==========================

install(TARGETS deploy_node
    DESTINATION lib/${PROJECT_NAME}
)

# Install Unitree SDK shared library alongside the executable
install(FILES ${UNITREE_SDK_LIB}
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
    OPTIONAL
)

install(DIRECTORY policy/
    DESTINATION share/${PROJECT_NAME}/policy
    OPTIONAL
)

# ========================== Package ==========================

ament_package()
