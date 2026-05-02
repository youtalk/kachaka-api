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

#include <gtest/gtest.h>

#include "kachaka_autoware_vehicle_interface/operation_mode_state_machine.hpp"

using kachaka_autoware_vehicle_interface::OperationMode;
using kachaka_autoware_vehicle_interface::OperationModeStateMachine;

TEST(OperationModeStateMachine, InitialStateIsStop)
{
  OperationModeStateMachine sm;
  EXPECT_EQ(sm.get_state(), OperationMode::STOP);
}

TEST(OperationModeStateMachine, RequestAutonomousFromStopSucceeds)
{
  OperationModeStateMachine sm;
  EXPECT_TRUE(sm.request_autonomous());
  EXPECT_EQ(sm.get_state(), OperationMode::AUTONOMOUS);
}

TEST(OperationModeStateMachine, RequestStopFromAutonomousSucceeds)
{
  OperationModeStateMachine sm;
  sm.request_autonomous();
  EXPECT_TRUE(sm.request_stop());
  EXPECT_EQ(sm.get_state(), OperationMode::STOP);
}

TEST(OperationModeStateMachine, RequestSameStateIsIdempotent)
{
  OperationModeStateMachine sm;
  EXPECT_TRUE(sm.request_stop());
  EXPECT_EQ(sm.get_state(), OperationMode::STOP);
  sm.request_autonomous();
  EXPECT_TRUE(sm.request_autonomous());
  EXPECT_EQ(sm.get_state(), OperationMode::AUTONOMOUS);
}
