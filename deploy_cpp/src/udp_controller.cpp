/**
 * @file udp_controller.cpp
 * @brief UDP teleop command receiver implementation.
 *
 * Runs a background thread that blocks on recvfrom() to receive
 * UdpCommand packets from the Python web relay server.
 */

#include "udp_controller.h"

#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>

namespace deploy {

UdpController::UdpController(int port) : port_(port) {
  // Create UDP socket
  sock_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (sock_fd_ < 0) {
    throw std::runtime_error("[UdpController] Failed to create UDP socket: " +
                             std::string(strerror(errno)));
  }

  // Allow address reuse
  int opt = 1;
  setsockopt(sock_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // Set receive timeout (1 second) so the thread can check running_ flag
  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  // Bind to 0.0.0.0:port
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port));

  if (::bind(sock_fd_, reinterpret_cast<struct sockaddr *>(&addr),
             sizeof(addr)) < 0) {
    ::close(sock_fd_);
    throw std::runtime_error("[UdpController] Failed to bind UDP port " +
                             std::to_string(port) + ": " +
                             std::string(strerror(errno)));
  }

  running_.store(true);
  thread_ = std::thread(&UdpController::listen_loop, this);

  std::cout << "[UdpController] Listening on UDP port " << port << std::endl;
}

UdpController::~UdpController() {
  running_.store(false);
  if (thread_.joinable()) {
    thread_.join();
  }
  if (sock_fd_ >= 0) {
    ::close(sock_fd_);
  }
}

UdpCommand UdpController::get_latest() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return latest_;
}

void UdpController::listen_loop() {
  uint8_t buf[64];
  struct sockaddr_in sender_addr{};
  socklen_t sender_len = sizeof(sender_addr);

  while (running_.load()) {
    ssize_t n = ::recvfrom(sock_fd_, buf, sizeof(buf), 0,
                           reinterpret_cast<struct sockaddr *>(&sender_addr),
                           &sender_len);

    if (n < 0) {
      // Timeout or interrupted — just retry
      continue;
    }

    if (static_cast<size_t>(n) != sizeof(UdpCommand)) {
      std::cerr << "[UdpController] Ignoring packet of size " << n
                << " (expected " << sizeof(UdpCommand) << ")" << std::endl;
      continue;
    }

    // Copy into latest_ under lock
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::memcpy(&latest_, buf, sizeof(UdpCommand));
    }
    has_data_.store(true, std::memory_order_release);
  }
}

} // namespace deploy
