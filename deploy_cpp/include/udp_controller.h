/**
 * @file udp_controller.h
 * @brief UDP-based teleop command receiver for mobile web control.
 *
 * Listens on a UDP port in a background thread and provides the
 * latest received command to the main control loop in a thread-safe
 * manner.
 *
 * Binary protocol (17 bytes, packed, little-endian):
 *   int32_t  mode      0=IDLE, 1=STAND_UP, 2=RL, 3=JOINT_DAMPING
 *   float    vx        normalized ratio [-1.0, 1.0]
 *   float    vy        normalized ratio [-1.0, 1.0]
 *   float    yaw       normalized ratio [-1.0, 1.0]
 *   uint8_t  e_stop    1 = emergency stop
 */
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

namespace deploy {

// ========================== UDP Command Struct ==========================

#pragma pack(push, 1)
struct UdpCommand {
  int32_t mode   = 0;     ///< 0-3 state
  float   vx     = 0.0f;  ///< vx ratio [-1, 1]
  float   vy     = 0.0f;  ///< vy ratio [-1, 1]
  float   yaw    = 0.0f;  ///< yaw ratio [-1, 1]
  uint8_t e_stop = 0;     ///< 1 = emergency stop
};
#pragma pack(pop)

static_assert(sizeof(UdpCommand) == 17, "UdpCommand must be 17 bytes packed");

// ========================== UDP Controller ==========================

class UdpController {
public:
  /**
   * @brief Construct and start listening on the given UDP port.
   * @param port UDP port to bind (e.g. 9870)
   */
  explicit UdpController(int port);

  ~UdpController();

  // Non-copyable
  UdpController(const UdpController &) = delete;
  UdpController &operator=(const UdpController &) = delete;

  /**
   * @brief Get the latest received command (thread-safe).
   * @return Copy of the most recent UdpCommand
   */
  UdpCommand get_latest() const;

  /**
   * @brief Whether at least one valid packet has been received.
   */
  bool has_data() const { return has_data_.load(std::memory_order_acquire); }

private:
  void listen_loop();

  int port_;
  int sock_fd_ = -1;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> has_data_{false};

  mutable std::mutex mutex_;
  UdpCommand latest_{};
};

} // namespace deploy
