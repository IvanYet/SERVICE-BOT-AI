#ifndef ARM_HARDWARE__ARM_HARDWARE_INTERFACE_HPP_
#define ARM_HARDWARE__ARM_HARDWARE_INTERFACE_HPP_

#include <functional>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace arm_hardware
{

// ros2_control SystemInterface for the 5-DOF arm + gripper, talking to arm_firmware.ino over USB serial.
// Firmware joint order: base_joint joint_1 joint_2 joint_3 joint_4 left_finger_joint.
class ArmHardwareInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(ArmHardwareInterface)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  static constexpr int kFwJoints = 6;

  std::vector<double> pos_;          // state, one per URDF joint
  std::vector<double> cmd_;          // command, one per URDF joint (unused for mimic joints)
  std::vector<int> fw_slot_;         // firmware slot of each URDF joint, -1 for mimic joints
  std::vector<int> mimic_src_;       // URDF joint index a mimic joint follows, -1 otherwise
  std::vector<double> mimic_mult_;
  int slot_joint_[kFwJoints] = {-1, -1, -1, -1, -1, -1};   // URDF joint index of each firmware slot

  std::string port_ = "/dev/ttyACM0";
  double homing_timeout_s_ = 180.0;
  int fd_ = -1;
  std::string rx_;
  int fw_state_ = -1;                // 0 not homed, 1 homing, 2 ready, 3 fault
  double fw_q_[kFwJoints] = {0, 0, 0, 0, 0, 0};

  bool open_port();
  void close_port();
  void send_line(const std::string & line);
  void poll_serial();
  void handle_line(const std::string & line);
  bool wait_for(const std::function<bool()> & done, double timeout_s);
  void update_positions();

  rclcpp::Logger logger_ = rclcpp::get_logger("ArmHardwareInterface");
  rclcpp::Clock clock_{RCL_STEADY_TIME};
};

}  // namespace arm_hardware

#endif  // ARM_HARDWARE__ARM_HARDWARE_INTERFACE_HPP_
