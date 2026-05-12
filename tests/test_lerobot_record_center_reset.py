from lerobot.scripts.lerobot_record import (
    _build_action_space_center_command,
    _return_to_action_space_center,
    _send_center_command_for_duration,
)


class _DummyRobot:
    action_features = {
        "left_joint_1.pos": float,
        "left_joint_2.pos": float,
        "left_gripper.pos": float,
        "right_joint_1.pos": float,
        "right_gripper.pos": float,
        "non_position": float,
    }

    def __init__(self):
        self.sent_actions = []

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        return action


class _Calibration:
    def __init__(self, homing_offset=0, drive_mode=0):
        self.homing_offset = homing_offset
        self.drive_mode = drive_mode


class _RampRobot(_DummyRobot):
    action_features = {"joint_1.pos": float}
    calibration = {"joint_1.pos": _Calibration(homing_offset=10000)}

    def _from_calibration_units(self, value):
        return float(value) / 1000.0

    def get_observation(self):
        return {"joint_1.pos": 16.0}


def test_build_action_space_center_command_excludes_grippers_by_default():
    command = _build_action_space_center_command(_DummyRobot())

    assert command == {
        "left_joint_1.pos": 0.0,
        "left_joint_2.pos": 0.0,
        "right_joint_1.pos": 0.0,
    }


def test_build_action_space_center_command_can_include_grippers():
    command = _build_action_space_center_command(_DummyRobot(), include_gripper=True)

    assert command == {
        "left_joint_1.pos": 0.0,
        "left_joint_2.pos": 0.0,
        "left_gripper.pos": 0.0,
        "right_joint_1.pos": 0.0,
        "right_gripper.pos": 0.0,
    }


def test_return_to_action_space_center_sends_center_command_once_for_zero_duration():
    robot = _DummyRobot()

    _return_to_action_space_center(
        robot,
        duration_s=0.0,
        control_hz=20.0,
        include_gripper=True,
        gripper_duration_s=0.0,
    )

    assert robot.sent_actions == [
        {
            "left_joint_1.pos": 0.0,
            "left_joint_2.pos": 0.0,
            "right_joint_1.pos": 0.0,
        },
        {
            "left_gripper.pos": 0.0,
            "right_gripper.pos": 0.0,
        },
    ]


def test_return_to_action_space_center_can_skip_grippers():
    robot = _DummyRobot()

    _return_to_action_space_center(
        robot,
        duration_s=0.0,
        control_hz=20.0,
        include_gripper=False,
        gripper_duration_s=0.0,
    )

    assert robot.sent_actions == [
        {
            "left_joint_1.pos": 0.0,
            "left_joint_2.pos": 0.0,
            "right_joint_1.pos": 0.0,
        }
    ]


def test_send_center_command_ramps_from_current_offset_to_zero():
    robot = _RampRobot()

    _send_center_command_for_duration(
        robot,
        {"joint_1.pos": 0.0},
        duration_s=1.0,
        control_hz=2.0,
        description="arm joints",
    )

    assert robot.sent_actions == [
        {"joint_1.pos": 3.0},
        {"joint_1.pos": 0.0},
    ]
