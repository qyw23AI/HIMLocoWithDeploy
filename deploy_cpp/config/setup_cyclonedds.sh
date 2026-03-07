#!/usr/bin/env bash
# =============================================================================
# setup_cyclonedds.sh — Switch ROS 2 to CycloneDDS with Shared Memory (SHM)
#
# Usage (source BEFORE launching any ROS 2 node):
#
#   cd deploy_cpp
#   source config/setup_cyclonedds.sh
#   ros2 launch deploy_cpp sim.launch.py
#
# Prerequisites:
#   sudo apt install ros-humble-rmw-cyclonedds-cpp
#
# To revert to default FastRTPS:
#   unset RMW_IMPLEMENTATION CYCLONEDDS_URI
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${SCRIPT_DIR}/cyclonedds.xml"

echo "[setup_cyclonedds] RMW_IMPLEMENTATION = ${RMW_IMPLEMENTATION}"
echo "[setup_cyclonedds] CYCLONEDDS_URI      = ${CYCLONEDDS_URI}"
echo "[setup_cyclonedds] Shared Memory (SHM) enabled for intra-host zero-copy."
