"""Vision jump requests are emitted once on an accepted platform transition."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from icn import IcnNode, MissionState
from track_phase import PhaseConfig, TrackPhase, TrackPhaseManager


class IcnJumpTest(unittest.TestCase):
    def make_node(self, enabled=True, motion=True, phase=TrackPhase.SEEK_PLATFORM):
        node = IcnNode.__new__(IcnNode)
        node._phase = TrackPhaseManager(
            PhaseConfig(platform_settle_seconds=0.0, platform_clear_frames=2,
                        first_corner_timeout_seconds=0.1),
            start_phase=phase, roundabout_enabled=False,
        )
        node.step_jump_enabled = enabled
        node.enable_motion = motion
        node.step_action = "climb"
        node.state = MissionState.LINE
        node._set_state = lambda state, reason: setattr(node, "state", state)
        node.requests = []
        node.warnings = []
        node.logs = []
        node._vision_step_pub = SimpleNamespace(
            publish=lambda msg: node.requests.append(msg.data),
            get_subscription_count=lambda: 1,
        )
        node._warn = node.warnings.append
        node.get_logger = lambda: SimpleNamespace(info=node.logs.append)
        return node

    def test_one_jump_per_lap_despite_repeated_detections(self):
        node = self.make_node()
        for lap in (1, 2):
            now = lap * 10.0
            node._enter_stair(now)
            self.assertEqual(node.state, MissionState.STAIR)
            for offset in (0.1, 0.2, 0.3):
                node._enter_stair(now + offset)
            self.assertEqual(node.requests, [False, True] * lap)
            node._phase.platform_action_done(now + 2.0)
            for offset in (2.1, 2.2):
                node._phase.observe_line(now + offset, True, 0.0, 1.0)
                node._phase.observe_step(now + offset, False)
            self.assertEqual(node._phase.phase, TrackPhase.SEEK_ZEBRA)
            node._phase.zebra_entered(now + 3.0)
            self.assertTrue(node._phase.zebra_passed(now + 4.0))
            self.assertTrue(node._phase.start_next_lap(now + 5.0))
            node._phase.observe_line(now + 6.0, True, 0.0, 1.0)
            self.assertEqual(node._phase.phase, TrackPhase.SEEK_PLATFORM)
        self.assertIn("第 2 圈", node.logs[-1])

    def test_dry_run_and_disabled_jump_send_nothing(self):
        for enabled, motion in ((True, False), (False, True), (False, False)):
            with self.subTest(enabled=enabled, motion=motion):
                node = self.make_node(enabled=enabled, motion=motion)
                node._enter_stair(1.0)
                self.assertEqual(node.state, MissionState.STAIR)
                self.assertEqual(node.requests, [])

    def test_wrong_phase_never_sends_jump(self):
        for phase in (TrackPhase.FIRST_CORNER, TrackPhase.SEEK_ZEBRA,
                      TrackPhase.CROSS_PLATFORM):
            with self.subTest(phase=phase):
                node = self.make_node(phase=phase)
                node._enter_stair(1.0)
                self.assertEqual(node.requests, [])
                self.assertEqual(node.state, MissionState.LINE)

    def test_missing_chassis_is_reported_without_repeated_requests(self):
        node = self.make_node()
        node._vision_step_pub.get_subscription_count = lambda: 0
        node._enter_stair(1.0)
        node._enter_stair(2.0)
        self.assertEqual(node.requests, [False, True])
        self.assertTrue(any("没有订阅者" in text for text in node.warnings))


if __name__ == "__main__":
    unittest.main()
