// Copyright 2026 Yutaka Kondo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kachaka_autoware_vehicle_interface/control_to_twist_converter.hpp"

#include <algorithm>
#include <cmath>

namespace kachaka_autoware_vehicle_interface
{

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
