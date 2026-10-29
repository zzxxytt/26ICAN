"""Track stage regressions; runnable without ROS or camera dependencies."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from track_phase import PhaseConfig, TrackElement, TrackPhase, TrackPhaseManager


class RoundaboutSwitchTest(unittest.TestCase):
    def test_off_opens_zebra_after_platform_clear_on_each_lap(self):
        transitions = []
        manager = TrackPhaseManager(
            PhaseConfig(platform_settle_seconds=0.0, platform_clear_frames=2,
                        first_corner_timeout_seconds=0.1),
            start_phase=TrackPhase.SEEK_PLATFORM,
            on_change=transitions.append,
            roundabout_enabled=False,
        )
        for lap in (1, 2):
            now = float(lap * 10)
            self.assertTrue(manager.begin_platform(now))
            manager.platform_action_done(now)
            self.assertEqual(manager.phase, TrackPhase.CROSS_PLATFORM)
            self.assertFalse(manager.permits(TrackElement.ZEBRA))
            for offset in (0.1, 0.2):
                manager.observe_line(now + offset, True, 0.0, 1.0)
                manager.observe_step(now + offset, False)
            # 不依赖圆环回调，平台清除的同一帧直接进入找斑马线。
            self.assertEqual(manager.phase, TrackPhase.SEEK_ZEBRA)
            self.assertFalse(any(t.new in (TrackPhase.SEEK_ROUNDABOUT, TrackPhase.ROUNDABOUT)
                                 for t in transitions))
            self.assertTrue(manager.may_trigger(TrackElement.ZEBRA))
            self.assertFalse(manager.permits(TrackElement.ROUNDABOUT))
            self.assertEqual(transitions[-1].reason, "roundabout_disabled")
            manager.zebra_entered(now + 0.4)
            self.assertTrue(manager.zebra_passed(now + 0.5))
            self.assertTrue(manager.start_next_lap(now + 0.6))
            manager.observe_line(now + 1.0, True, 0.0, 1.0)
            self.assertEqual(manager.phase, TrackPhase.SEEK_PLATFORM)

    def test_off_skips_either_ring_start_phase_immediately(self):
        for phase in (TrackPhase.SEEK_ROUNDABOUT, TrackPhase.ROUNDABOUT):
            with self.subTest(phase=phase):
                manager = TrackPhaseManager(start_phase=phase)
                manager.observe_roundabout(1.0, mode="off", state="FOLLOW",
                                          junction=True)
                self.assertEqual(manager.phase, TrackPhase.SEEK_ZEBRA)
                self.assertTrue(manager.may_trigger(TrackElement.ZEBRA))

    def test_disabled_start_phase_never_reports_ring(self):
        for phase in (TrackPhase.SEEK_ROUNDABOUT, TrackPhase.ROUNDABOUT):
            manager = TrackPhaseManager(start_phase=phase, roundabout_enabled=False)
            self.assertEqual(manager.phase, TrackPhase.SEEK_ZEBRA)
            self.assertFalse(manager.permits(TrackElement.ROUNDABOUT))

    def test_reenable_does_not_rewind_current_lap(self):
        manager = TrackPhaseManager(start_phase=TrackPhase.ROUNDABOUT)
        manager.set_roundabout_enabled(False, 1.0)
        manager.set_roundabout_enabled(True, 2.0)
        self.assertEqual(manager.phase, TrackPhase.SEEK_ZEBRA)
        self.assertTrue(manager.roundabout_enabled)

    def test_off_does_not_skip_other_stages(self):
        for phase in (TrackPhase.FIRST_CORNER, TrackPhase.SEEK_PLATFORM,
                      TrackPhase.CROSS_PLATFORM, TrackPhase.SEEK_ZEBRA,
                      TrackPhase.LAP_COMPLETE):
            with self.subTest(phase=phase):
                manager = TrackPhaseManager(start_phase=phase)
                manager.observe_roundabout(1.0, mode="off")
                self.assertEqual(manager.phase, phase)

    def test_enabled_modes_still_require_ring_evidence(self):
        for mode in ("observe", "control"):
            with self.subTest(mode=mode):
                manager = TrackPhaseManager(
                    PhaseConfig(roundabout_min_seconds=0.0, roundabout_clear_frames=2),
                    start_phase=TrackPhase.SEEK_ROUNDABOUT,
                )
                manager.observe_roundabout(1.0, mode=mode)
                self.assertEqual(manager.phase, TrackPhase.SEEK_ROUNDABOUT)
                self.assertFalse(manager.permits(TrackElement.ZEBRA))
                manager.observe_roundabout(2.0, mode=mode, state="ENTER")
                self.assertEqual(manager.phase, TrackPhase.ROUNDABOUT)
                for now in (3.0, 4.0):
                    manager.observe_roundabout(now, mode=mode, completed=1)
                self.assertEqual(manager.phase, TrackPhase.SEEK_ZEBRA)


if __name__ == "__main__":
    unittest.main()
