import ast
import unittest
from pathlib import Path

import debug_step8_24d_wrist_raw_clean as harness


class FakeCalibration:
    def __init__(self, low, high):
        self.range_min = low
        self.range_max = high


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeBus:
    def __init__(self):
        self.calibration = {
            "shoulder_pan": FakeCalibration(920, 3444),
            "shoulder_lift": FakeCalibration(819, 3217),
            "elbow_flex": FakeCalibration(869, 3087),
            "wrist_flex": FakeCalibration(906, 3206),
            "wrist_roll": FakeCalibration(0, 4095),
            "gripper": FakeCalibration(600, 3000),
        }
        self.values = {
            "Torque_Enable": dict.fromkeys(harness.ALL_MOTOR_NAMES, 1),
            "P_Coefficient": dict.fromkeys(harness.ALL_MOTOR_NAMES, 64),
            "I_Coefficient": dict.fromkeys(harness.ALL_MOTOR_NAMES, 0),
            "D_Coefficient": dict.fromkeys(harness.ALL_MOTOR_NAMES, 32),
            "Present_Position": {
                "shoulder_pan": 2000, "shoulder_lift": 1900, "elbow_flex": 2100,
                "wrist_flex": 2500, "wrist_roll": 2048, "gripper": 1500,
            },
            "Goal_Position": {
                "shoulder_pan": 2000, "shoulder_lift": 1900, "elbow_flex": 2100,
                "wrist_flex": 2504, "wrist_roll": 2048, "gripper": 1500,
            },
            "Present_Temperature": dict.fromkeys(harness.ALL_MOTOR_NAMES, 30),
            "Present_Load": dict.fromkeys(harness.ALL_MOTOR_NAMES, 0),
            "Status": dict.fromkeys(harness.ALL_MOTOR_NAMES, 0),
        }
        self.writes = []
        self.fail_write = False
        self.readback_mismatch = False

    def sync_read(self, register_name, motors=None, *, normalize=True):
        if normalize:
            raise AssertionError("test requires explicit normalize=False")
        result = dict(self.values[register_name])
        if register_name == "Goal_Position" and self.readback_mismatch:
            result["wrist_flex"] += 1
        return result

    def sync_write(self, register_name, values, *, normalize=True):
        self.writes.append((register_name, dict(values), normalize))
        if self.fail_write:
            raise OSError("injected communication exception")
        if register_name != "Goal_Position" or set(values) != {"wrist_flex"} or normalize:
            raise AssertionError("unauthorized fake-bus write")
        self.values[register_name].update(values)


class FakeInitializationBus(FakeBus):
    def sync_write(self, register_name, values, *, normalize=True):
        self.writes.append((register_name, dict(values), normalize))
        if self.fail_write:
            raise OSError("injected communication exception")
        if set(values) != set(harness.ARM_INITIALIZATION_MOTOR_NAMES) or "gripper" in values or normalize:
            raise AssertionError("unauthorized initialization write")
        if register_name not in {"Torque_Enable", "Goal_Position", "P_Coefficient", "I_Coefficient", "D_Coefficient"}:
            raise AssertionError("unauthorized initialization register")
        self.values[register_name].update(values)


class FakeReturnFailureBus(FakeBus):
    def sync_write(self, register_name, values, *, normalize=True):
        super().sync_write(register_name, values, normalize=normalize)
        if register_name == "Goal_Position" and values["wrist_flex"] != 2500:
            self.values["Present_Position"]["wrist_flex"] = 2503


class FakeThreeJointBus(FakeBus):
    def sync_write(self, register_name, values, *, normalize=True):
        self.writes.append((register_name, dict(values), normalize))
        if register_name != "Goal_Position" or set(values) - set(harness.THREE_JOINT_SWEEP_NAMES) or len(values) != 1 or normalize:
            raise AssertionError("unauthorized three-joint write")
        self.values[register_name].update(values)


class FakeStartPoseBus(FakeThreeJointBus):
    def __init__(self):
        super().__init__()
        self.normalized_reads = 0
        self.normalized_present = dict.fromkeys(harness.INFERENCE_START_ARM_JOINT_NAMES, 0.0)
        self.normalized_tracking_offset = {
            "shoulder_pan": 0.1,
            "shoulder_lift": -0.1,
            "elbow_flex": 1.669,
            "wrist_flex": 0.1,
            "wrist_roll": -0.1,
        }

    def sync_read(self, register_name, motors=None, *, normalize=True):
        if normalize and register_name == "Present_Position":
            self.normalized_reads += 1
            if self.normalized_reads == 1:
                return dict.fromkeys(harness.INFERENCE_START_ARM_JOINT_NAMES, 0.0)
            return dict(self.normalized_present)
        return super().sync_read(register_name, motors, normalize=normalize)

    def sync_write(self, register_name, values, *, normalize=True):
        if normalize:
            self.writes.append((register_name, dict(values), normalize))
            if register_name != "Goal_Position" or set(values) != set(harness.INFERENCE_START_ARM_JOINT_NAMES):
                raise AssertionError("unauthorized start-pose write")
            self.normalized_present = {
                joint_name: values[joint_name] + self.normalized_tracking_offset[joint_name]
                for joint_name in harness.INFERENCE_START_ARM_JOINT_NAMES
            }
            return
        return super().sync_write(register_name, values, normalize=normalize)


def preflight(bus):
    clock = FakeClock()
    return harness.preflight_read_only(bus, monotonic_fn=clock.monotonic, sleep_fn=clock.sleep)


class TestStep824DCleanHarness(unittest.TestCase):
    def test_static_source_rules(self):
        source_path = Path(harness.__file__)
        source = source_path.read_text()
        tree = ast.parse(source)
        self.assertNotIn("SOFollower.connect", source)
        self.assertNotIn("follower.connect", source)
        self.assertNotIn("follower.configure", source)
        self.assertNotIn("follower.disconnect", source)
        sync_writes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sync_write"
        ]
        direct_writes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write"
        ]
        self.assertEqual(direct_writes, [])
        # Seven sites: six raw diagnostic writers and one calibrated-degree start-pose writer.
        self.assertEqual(len(sync_writes), 7)
        normalized_true_count = 0
        for call in sync_writes:
            self.assertEqual(call.keywords[0].arg, "normalize")
            if call.keywords[0].value.value:
                normalized_true_count += 1
            else:
                self.assertEqual(call.keywords[0].value.value, False)
        self.assertEqual(normalized_true_count, 1)
        self.assertIn("class ArmKnownStateInitializer", source)
        self.assertIn('assert "gripper" not in motors', source)

    def test_default_args_cannot_arm_writes(self):
        args = harness.parse_args(["--port", "mock", "--calibration-json", "mock.json"])
        self.assertFalse(args.execute_micro_test)
        self.assertNotEqual(args.arm_token, harness.ARM_TOKEN)

    def test_mock_success_exact_write_trace(self):
        bus = FakeBus()
        report = preflight(bus)
        audit = harness.WriteAudit()
        harness.run_micro_test_mockable(bus, report, audit)
        self.assertEqual(
            bus.writes,
            [
                ("Goal_Position", {"wrist_flex": 2505}, False),
                ("Goal_Position", {"wrist_flex": 2504}, False),
                ("Goal_Position", {"wrist_flex": 2503}, False),
                ("Goal_Position", {"wrist_flex": 2504}, False),
                ("Goal_Position", {"wrist_flex": 2506}, False),
                ("Goal_Position", {"wrist_flex": 2504}, False),
                ("Goal_Position", {"wrist_flex": 2502}, False),
                ("Goal_Position", {"wrist_flex": 2504}, False),
            ],
        )
        self.assertEqual(audit.goal_write_count_wrist, 8)
        audit.assert_forbidden_counters_zero()

    def test_mock_arm_initialization_exact_trace_and_gripper_exclusion(self):
        bus = FakeInitializationBus()
        audit = harness.WriteAudit()
        harness.ArmKnownStateInitializer(bus, audit).initialize_known_arm_state(sleep_fn=lambda _: None)
        arm = set(harness.ARM_INITIALIZATION_MOTOR_NAMES)
        present = {
            "shoulder_pan": 2000, "shoulder_lift": 1900, "elbow_flex": 2100,
            "wrist_flex": 2500, "wrist_roll": 2048,
        }
        self.assertEqual(
            bus.writes,
            [
                ("Torque_Enable", dict.fromkeys(harness.ARM_INITIALIZATION_MOTOR_NAMES, 0), False),
                ("Goal_Position", present, False),
                ("P_Coefficient", dict.fromkeys(harness.ARM_INITIALIZATION_MOTOR_NAMES, 64), False),
                ("I_Coefficient", dict.fromkeys(harness.ARM_INITIALIZATION_MOTOR_NAMES, 0), False),
                ("D_Coefficient", dict.fromkeys(harness.ARM_INITIALIZATION_MOTOR_NAMES, 32), False),
                ("Torque_Enable", dict.fromkeys(harness.ARM_INITIALIZATION_MOTOR_NAMES, 1), False),
            ],
        )
        self.assertTrue(all(set(values) == arm and "gripper" not in values for _, values, _ in bus.writes))
        self.assertEqual(audit.goal_write_count_gripper, 0)
        self.assertEqual(audit.config_write_count, 0)

    def test_mock_sweep_exact_sequence_restores_frozen_g0_and_wrist_only(self):
        bus = FakeBus()
        bus.values["Goal_Position"]["wrist_flex"] = 2500
        clock = FakeClock()
        audit = harness.WriteAudit()
        results = harness.TrueRawWristSweep(bus, audit).run(
            monotonic_fn=clock.monotonic, sleep_fn=clock.sleep
        )
        self.assertEqual([result.requested_delta_raw_step for result in results], [1, -1, 2, -2, 4, -4, 8, -8])
        self.assertEqual(
            bus.writes,
            [
                ("Goal_Position", {"wrist_flex": 2501}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2499}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2502}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2498}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2504}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2496}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2508}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
                ("Goal_Position", {"wrist_flex": 2492}, False), ("Goal_Position", {"wrist_flex": 2500}, False),
            ],
        )
        self.assertTrue(all(set(values) == {"wrist_flex"} and not normalize for _, values, normalize in bus.writes))
        audit.assert_forbidden_counters_zero()

    def test_sweep_range_gate_has_zero_writes(self):
        bus = FakeBus()
        bus.values["Goal_Position"]["wrist_flex"] = 3200
        bus.values["Present_Position"]["wrist_flex"] = 3200
        clock = FakeClock()
        with self.assertRaises(RuntimeError):
            harness.TrueRawWristSweep(bus, harness.WriteAudit()).run(
                monotonic_fn=clock.monotonic, sleep_fn=clock.sleep
            )
        self.assertEqual(bus.writes, [])

    def test_sweep_return_timeout_stops_before_next_stimulus(self):
        bus = FakeReturnFailureBus()
        bus.values["Goal_Position"]["wrist_flex"] = 2500
        clock = FakeClock()
        with self.assertRaises(RuntimeError):
            harness.TrueRawWristSweep(bus, harness.WriteAudit()).run(
                monotonic_fn=clock.monotonic, sleep_fn=clock.sleep
            )
        self.assertEqual(
            bus.writes,
            [
                ("Goal_Position", {"wrist_flex": 2501}, False),
                ("Goal_Position", {"wrist_flex": 2500}, False),
            ],
        )

    def test_three_joint_sweep_isolated_exact_order_and_restores_each_g0(self):
        bus = FakeThreeJointBus()
        bus.values["Goal_Position"]["wrist_flex"] = 2500
        clock = FakeClock()
        results = harness.ThreeJointRawSweep(bus, harness.WriteAudit()).run(
            monotonic_fn=clock.monotonic, sleep_fn=clock.sleep
        )
        self.assertEqual(list(results), list(harness.THREE_JOINT_SWEEP_NAMES))
        self.assertEqual(len(bus.writes), 48)
        for index, joint_name in enumerate(harness.THREE_JOINT_SWEEP_NAMES):
            g0 = {"shoulder_lift": 1900, "elbow_flex": 2100, "wrist_flex": 2500}[joint_name]
            writes = bus.writes[index * 16:(index + 1) * 16]
            self.assertEqual([next(iter(values)) for _, values, _ in writes], [joint_name] * 16)
            self.assertEqual([next(iter(values.values())) for _, values, _ in writes], [
                g0+1,g0,g0-1,g0,g0+2,g0,g0-2,g0,g0+4,g0,g0-4,g0,g0+8,g0,g0-8,g0
            ])
        self.assertTrue(all("gripper" not in values and not normalize for _, values, normalize in bus.writes))

    def test_visible_response_uses_15_and_20_degrees_and_restores_g0(self):
        self.assertEqual(harness.VISIBLE_RESPONSE_DEGREES, (1.5, -1.5, 2.0, -2.0))
        self.assertEqual([harness.VisibleJointResponseTest.degrees_to_raw_step_delta(d) for d in harness.VISIBLE_RESPONSE_DEGREES], [17, -17, 23, -23])
        bus = FakeThreeJointBus()
        bus.values["Goal_Position"]["wrist_flex"] = 2500
        clock = FakeClock()
        results = harness.VisibleJointResponseTest(bus, harness.WriteAudit()).run(
            monotonic_fn=clock.monotonic, sleep_fn=clock.sleep
        )
        self.assertEqual(list(results), list(harness.THREE_JOINT_SWEEP_NAMES))
        self.assertEqual(len(bus.writes), 24)
        for index, joint_name in enumerate(harness.THREE_JOINT_SWEEP_NAMES):
            g0 = {"shoulder_lift": 1900, "elbow_flex": 2100, "wrist_flex": 2500}[joint_name]
            writes = bus.writes[index * 8:(index + 1) * 8]
            self.assertEqual([next(iter(v.values())) for _, v, _ in writes], [g0+17,g0,g0-17,g0,g0+23,g0,g0-23,g0])
            self.assertTrue(all(set(v) == {joint_name} and not n for _,v,n in writes))

    def test_visible_response_skips_only_calibration_invalid_direction(self):
        bus = FakeThreeJointBus()
        bus.values["Goal_Position"]["elbow_flex"] = 3070
        bus.values["Present_Position"]["elbow_flex"] = 3070
        bus.values["Goal_Position"]["wrist_flex"] = 2500
        clock = FakeClock()
        results = harness.VisibleJointResponseTest(bus, harness.WriteAudit()).run(
            monotonic_fn=clock.monotonic, sleep_fn=clock.sleep
        )
        self.assertEqual([result.requested_delta_raw_step for result in results["elbow_flex"]], [-17, -23])
        self.assertEqual(len(results["shoulder_lift"]), 4)
        self.assertEqual(len(results["wrist_flex"]), 4)

    def test_start_pose_values_order_smooth_interpolation_and_no_gripper(self):
        self.assertEqual(
            harness.INFERENCE_START_ARM_JOINT_NAMES,
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        )
        self.assertEqual(harness.INFERENCE_START_PHYSICAL_DEGREES, (-6.945, -81.462, 65.760, 46.132, 1.538))
        bus = FakeStartPoseBus(); audit = harness.WriteAudit(); clock = FakeClock()
        harness.InferenceStartPoseMove(bus, audit).move_and_settle(monotonic_fn=clock.monotonic, sleep_fn=clock.sleep)
        interpolation_writes = int(harness.START_POSE_DURATION_SECONDS * harness.START_POSE_UPDATE_HZ)
        trim_writes = bus.writes[interpolation_writes:]
        self.assertGreater(len(trim_writes), 0)
        self.assertLessEqual(len(trim_writes), harness.START_POSE_TRIM_MAX_ITERATIONS)
        self.assertEqual(len(bus.writes), interpolation_writes + len(trim_writes))
        self.assertTrue(all(set(values) == set(harness.INFERENCE_START_ARM_JOINT_NAMES) and "gripper" not in values and normalize for _,values,normalize in bus.writes))
        self.assertNotEqual(bus.writes[0][1], bus.writes[-1][1])
        nominal_goal = dict(zip(harness.INFERENCE_START_ARM_JOINT_NAMES, harness.INFERENCE_START_PHYSICAL_DEGREES))
        for _, goal, _ in trim_writes:
            for joint_name in harness.INFERENCE_START_ARM_JOINT_NAMES:
                self.assertLessEqual(abs(goal[joint_name] - nominal_goal[joint_name]), 1.5)
        for previous, current in zip(trim_writes, trim_writes[1:]):
            for joint_name in harness.INFERENCE_START_ARM_JOINT_NAMES:
                self.assertLessEqual(abs(current[1][joint_name] - previous[1][joint_name]), 0.75)
        self.assertEqual(audit.goal_write_count_gripper, 0)

    def test_initialization_goal_readback_mismatch_does_not_reenable_or_recover(self):
        bus = FakeInitializationBus()
        bus.readback_mismatch = True
        audit = harness.WriteAudit()
        with self.assertRaises(RuntimeError):
            harness.ArmKnownStateInitializer(bus, audit).initialize_known_arm_state(sleep_fn=lambda _: None)
        self.assertEqual([register for register, _, _ in bus.writes], [
            "Torque_Enable", "Goal_Position", "P_Coefficient", "I_Coefficient", "D_Coefficient"
        ])
        self.assertNotIn(("Torque_Enable", dict.fromkeys(harness.ARM_INITIALIZATION_MOTOR_NAMES, 1), False), bus.writes)

    def test_readback_mismatch_stops_without_recovery_write(self):
        bus = FakeBus()
        report = preflight(bus)
        bus.readback_mismatch = True
        audit = harness.WriteAudit()
        with self.assertRaises(RuntimeError):
            harness.run_micro_test_mockable(bus, report, audit)
        self.assertEqual(bus.writes, [("Goal_Position", {"wrist_flex": 2505}, False)])
        audit.assert_forbidden_counters_zero()

    def test_out_of_range_target_prevents_any_write(self):
        bus = FakeBus()
        bus.values["Goal_Position"]["wrist_flex"] = 3205
        bus.values["Present_Position"]["wrist_flex"] = 3205
        with self.assertRaises(RuntimeError):
            preflight(bus)
        self.assertEqual(bus.writes, [])

    def test_excessive_gap_is_reported_in_preflight_but_refused_when_armed(self):
        bus = FakeBus()
        bus.values["Goal_Position"]["wrist_flex"] = 2520
        report = preflight(bus)
        with self.assertRaises(RuntimeError):
            harness.validate_armed_execution_gate(
                report, harness.PROPOSED_MAX_WRIST_GOAL_PRESENT_GAP_RAW_STEP
            )
        self.assertEqual(bus.writes, [])

    def test_wrong_arm_token_prevents_state_machine_call(self):
        args = harness.parse_args(["--port", "mock", "--calibration-json", "mock.json", "--execute-micro-test"])
        self.assertNotEqual(args.arm_token, harness.ARM_TOKEN)

    def test_moving_baseline_prevents_any_write(self):
        bus = FakeBus()
        original_read = bus.sync_read
        reads = {"n": 0}
        def moving_read(register_name, motors=None, *, normalize=True):
            result = original_read(register_name, motors, normalize=normalize)
            if register_name == "Present_Position":
                reads["n"] += 1
                if reads["n"] >= 2:
                    result["shoulder_lift"] += 3
            return result
        bus.sync_read = moving_read
        with self.assertRaises(RuntimeError):
            preflight(bus)
        self.assertEqual(bus.writes, [])

    def test_pid_mismatch_torque_off_and_gripper_status_each_prevent_writes(self):
        for register_name, motor_name, value in [
            ("P_Coefficient", "wrist_flex", 16),
            ("Torque_Enable", "wrist_flex", 0),
            ("Status", "gripper", 1),
        ]:
            with self.subTest(register=register_name):
                bus = FakeBus()
                bus.values[register_name][motor_name] = value
                with self.assertRaises(RuntimeError):
                    preflight(bus)
                self.assertEqual(bus.writes, [])

    def test_write_communication_exception_has_no_recovery_write(self):
        bus = FakeBus()
        report = preflight(bus)
        bus.fail_write = True
        audit = harness.WriteAudit()
        with self.assertRaises(OSError):
            harness.run_micro_test_mockable(bus, report, audit)
        self.assertEqual(len(bus.writes), 1)
        self.assertEqual(bus.writes[0], ("Goal_Position", {"wrist_flex": 2505}, False))
        audit.assert_forbidden_counters_zero()


if __name__ == "__main__":
    unittest.main(verbosity=2)
