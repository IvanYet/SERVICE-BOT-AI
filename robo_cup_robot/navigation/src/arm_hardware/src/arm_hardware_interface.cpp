#include "arm_hardware/arm_hardware_interface.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <thread>

#include "pluginlib/class_list_macros.hpp"

namespace arm_hardware
{
using hardware_interface::CallbackReturn;
using hardware_interface::return_type;

CallbackReturn ArmHardwareInterface::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  const auto & hp = info_.hardware_parameters;
  if (hp.count("serial_port")) port_ = hp.at("serial_port");
  if (hp.count("homing_timeout")) homing_timeout_s_ = std::stod(hp.at("homing_timeout"));

  static const std::map<std::string, int> slots = {
    {"base_joint", 0}, {"joint_1", 1}, {"joint_2", 2}, {"joint_3", 3}, {"joint_4", 4}, {"left_finger_joint", 5}};

  const size_t n = info_.joints.size();
  pos_.assign(n, 0.0);
  cmd_.assign(n, std::numeric_limits<double>::quiet_NaN());
  fw_slot_.assign(n, -1);
  mimic_src_.assign(n, -1);
  mimic_mult_.assign(n, 1.0);

  for (size_t j = 0; j < n; ++j) {
    const auto & joint = info_.joints[j];
    auto it = slots.find(joint.name);
    if (it != slots.end()) {
      fw_slot_[j] = it->second;
      slot_joint_[it->second] = static_cast<int>(j);
      continue;
    }
    auto mp = joint.parameters.find("mimic");
    if (mp != joint.parameters.end()) {
      for (size_t k = 0; k < n; ++k) {
        if (info_.joints[k].name == mp->second) mimic_src_[j] = static_cast<int>(k);
      }
      auto mm = joint.parameters.find("multiplier");
      if (mm != joint.parameters.end()) mimic_mult_[j] = std::stod(mm->second);
      if (mimic_src_[j] < 0) {
        RCLCPP_ERROR(logger_, "Joint '%s' mimics unknown joint '%s'", joint.name.c_str(), mp->second.c_str());
        return CallbackReturn::ERROR;
      }
      continue;
    }
    RCLCPP_ERROR(logger_, "Joint '%s' is not handled by the arm firmware", joint.name.c_str());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> ArmHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> out;
  for (size_t j = 0; j < info_.joints.size(); ++j) {
    out.emplace_back(info_.joints[j].name, hardware_interface::HW_IF_POSITION, &pos_[j]);
  }
  return out;
}

std::vector<hardware_interface::CommandInterface> ArmHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  for (size_t j = 0; j < info_.joints.size(); ++j) {
    if (mimic_src_[j] >= 0) continue;   // mimic joints are not commanded
    out.emplace_back(info_.joints[j].name, hardware_interface::HW_IF_POSITION, &cmd_[j]);
  }
  return out;
}

CallbackReturn ArmHardwareInterface::on_configure(const rclcpp_lifecycle::State &)
{
  if (!open_port()) {
    RCLCPP_ERROR(logger_, "Cannot open %s: %s", port_.c_str(), std::strerror(errno));
    return CallbackReturn::ERROR;
  }
  fw_state_ = -1;
  // The ESP32-S3 native USB port sometimes stays silent after the previous program closed it.
  // Re-opening it with a fresh DTR edge (what a serial monitor does) wakes it up, so try a few times.
  bool got = false;
  for (int attempt = 0; attempt < 3 && !got; ++attempt) {
    if (attempt > 0) {
      RCLCPP_WARN(logger_, "No data yet from %s, re-opening the port (attempt %d of 3)", port_.c_str(), attempt + 1);
      close_port();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      if (!open_port()) {
        RCLCPP_ERROR(logger_, "Cannot re-open %s: %s", port_.c_str(), std::strerror(errno));
        return CallbackReturn::ERROR;
      }
    }
    send_line("");   // an empty line is ignored by the firmware, it only gets data moving
    got = wait_for([this] { return fw_state_ >= 0; }, 3.0);
  }
  if (!got) {
    RCLCPP_ERROR(logger_, "No data from the arm on %s. Is the firmware running, and is the Arduino serial monitor closed? "
                 "Pressing RST on the ESP32 and launching again usually clears this.", port_.c_str());
    close_port();
    return CallbackReturn::ERROR;
  }
  RCLCPP_INFO(logger_, "Connected to the arm on %s", port_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn ArmHardwareInterface::on_activate(const rclcpp_lifecycle::State &)
{
  if (fw_state_ != 2) {
    RCLCPP_INFO(logger_, "Homing the arm (up to %.0f s). Keep clear of the arm.", homing_timeout_s_);
    if (fw_state_ == 3) send_line("R");
    send_line("H");
    if (!wait_for([this] { return fw_state_ == 1; }, 3.0)) {
      RCLCPP_ERROR(logger_, "The arm did not start homing");
      return CallbackReturn::ERROR;
    }
    if (!wait_for([this] { return fw_state_ == 2 || fw_state_ == 3; }, homing_timeout_s_) || fw_state_ != 2) {
      RCLCPP_ERROR(logger_, "Homing failed or timed out (see the 'arm:' messages above)");
      send_line("X");
      return CallbackReturn::ERROR;
    }
  } else {
    RCLCPP_INFO(logger_, "The arm is already homed");
  }
  update_positions();
  for (size_t j = 0; j < cmd_.size(); ++j) cmd_[j] = pos_[j];
  RCLCPP_INFO(logger_, "Arm ready");
  return CallbackReturn::SUCCESS;
}

CallbackReturn ArmHardwareInterface::on_deactivate(const rclcpp_lifecycle::State &)
{
  if (fd_ >= 0) send_line("X");
  close_port();
  return CallbackReturn::SUCCESS;
}

return_type ArmHardwareInterface::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  poll_serial();
  update_positions();
  if (fw_state_ == 3) {
    RCLCPP_ERROR_THROTTLE(logger_, clock_, 3000,
                          "The arm is in a fault state and ignores commands. Fix the cause, then restart to re-home.");
  }
  return return_type::OK;
}

return_type ArmHardwareInterface::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (fw_state_ != 2) return return_type::OK;
  double v[kFwJoints];
  for (int s = 0; s < kFwJoints; ++s) {
    const int j = slot_joint_[s];
    v[s] = (j >= 0) ? cmd_[j] : fw_q_[s];
    if (!std::isfinite(v[s])) return return_type::OK;
  }
  char buf[160];
  std::snprintf(buf, sizeof(buf), "P %.5f %.5f %.5f %.5f %.5f %.5f", v[0], v[1], v[2], v[3], v[4], v[5]);
  send_line(buf);
  return return_type::OK;
}

void ArmHardwareInterface::update_positions()
{
  for (size_t j = 0; j < pos_.size(); ++j) {
    if (fw_slot_[j] >= 0) pos_[j] = fw_q_[fw_slot_[j]];
  }
  for (size_t j = 0; j < pos_.size(); ++j) {
    if (mimic_src_[j] >= 0) pos_[j] = mimic_mult_[j] * pos_[mimic_src_[j]];
  }
}

// Find the live ESP32-S3 native USB port (USB id 303a:1001) by looking in sysfs. A port whose
// board has disconnected disappears from sysfs, so this never picks a dead leftover node.
static std::string find_esp32_port()
{
  namespace fs = std::filesystem;
  std::error_code ec;
  std::vector<std::string> names;
  for (const auto & e : fs::directory_iterator("/sys/class/tty", ec)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("ttyACM", 0) == 0) names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  for (const auto & name : names) {
    // /sys/class/tty/ttyACMx/device -> the USB interface; its parent holds idVendor
    const fs::path usb_dev = fs::canonical("/sys/class/tty/" + name + "/device", ec).parent_path();
    if (ec) continue;
    std::ifstream vf(usb_dev / "idVendor");
    std::string vid;
    if (vf >> vid && vid == "303a") return "/dev/" + name;
  }
  return "";
}

bool ArmHardwareInterface::open_port()
{
  std::string dev = port_;
  if (port_ == "auto" || ::access(port_.c_str(), F_OK) != 0) {
    // The board may be re-attaching (usbipd --auto-attach takes a few seconds): wait up to 10 s.
    std::string found = find_esp32_port();
    for (int i = 0; i < 20 && found.empty(); ++i) {
      if (i == 0) RCLCPP_WARN(logger_, "No ESP32 USB port found yet, waiting up to 10 s for it to be attached...");
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      found = find_esp32_port();
    }
    if (found.empty()) {
      RCLCPP_ERROR(logger_, "No ESP32 USB port found. In Windows PowerShell run: usbipd attach --wsl --busid <BUSID> "
                   "(see 'usbipd list'), then 'ls /dev/ttyACM*' in Ubuntu should show it.");
      errno = ENODEV;
      return false;
    }
    if (found != port_) RCLCPP_INFO(logger_, "Using ESP32 port %s", found.c_str());
    dev = found;
  }
  fd_ = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) return false;
  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) { close_port(); return false; }
  cfmakeraw(&tty);
  cfsetispeed(&tty, B115200);   // ignored by the ESP32-S3 native USB port, but harmless
  cfsetospeed(&tty, B115200);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~HUPCL;        // do not drop DTR on close
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;
  if (tcsetattr(fd_, TCSANOW, &tty) != 0) { close_port(); return false; }
  // Give the port a clean DTR off->on edge, like a serial monitor does when it connects.
  // RTS is kept low throughout: on the ESP32-S3 it is RTS (with DTR low) that resets the chip.
  int rts = TIOCM_RTS, dtr = TIOCM_DTR;
  ioctl(fd_, TIOCMBIC, &rts);
  ioctl(fd_, TIOCMBIC, &dtr);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ioctl(fd_, TIOCMBIS, &dtr);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  tcflush(fd_, TCIOFLUSH);
  rx_.clear();
  return true;
}

void ArmHardwareInterface::close_port()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void ArmHardwareInterface::send_line(const std::string & line)
{
  if (fd_ < 0) return;
  const std::string out = line + "\n";
  ssize_t n = ::write(fd_, out.data(), out.size());
  if (n < 0 && errno != EAGAIN) {
    RCLCPP_ERROR_THROTTLE(logger_, clock_, 3000, "Serial write failed: %s", std::strerror(errno));
  }
}

void ArmHardwareInterface::poll_serial()
{
  if (fd_ < 0) return;
  char buf[512];
  while (true) {
    ssize_t n = ::read(fd_, buf, sizeof(buf));
    if (n <= 0) break;
    rx_.append(buf, static_cast<size_t>(n));
  }
  size_t nl;
  while ((nl = rx_.find('\n')) != std::string::npos) {
    std::string line = rx_.substr(0, nl);
    rx_.erase(0, nl + 1);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) handle_line(line);
  }
  if (rx_.size() > 8192) rx_.clear();
}

void ArmHardwareInterface::handle_line(const std::string & line)
{
  const char * text = line.size() > 2 ? line.c_str() + 2 : "";
  switch (line[0]) {
    case 'J': {
      int st;
      double q[kFwJoints];
      int used = 0;
      // Require exactly 7 fields and nothing after them, so a line that lost bytes and merged
      // with the next one on the serial link is dropped instead of giving mixed-up positions.
      if (std::sscanf(line.c_str(), "J %d %lf %lf %lf %lf %lf %lf %n", &st, &q[0], &q[1], &q[2], &q[3], &q[4], &q[5], &used) == 7 &&
          used == static_cast<int>(line.size()) && st >= 0 && st <= 3) {
        fw_state_ = st;
        for (int i = 0; i < kFwJoints; ++i) fw_q_[i] = q[i];
      }
      break;
    }
    case 'F':
      RCLCPP_ERROR(logger_, "arm: %s", text);
      break;
    case 'I':
      RCLCPP_INFO(logger_, "arm: %s", text);
      break;
    default:
      break;
  }
}

bool ArmHardwareInterface::wait_for(const std::function<bool()> & done, double timeout_s)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    poll_serial();
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

}  // namespace arm_hardware

PLUGINLIB_EXPORT_CLASS(arm_hardware::ArmHardwareInterface, hardware_interface::SystemInterface)
