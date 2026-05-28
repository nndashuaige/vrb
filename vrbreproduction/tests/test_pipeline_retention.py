import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vrbreproduction.pipeline_retention import (
    StageResult,
    build_event_records,
    summarize_stage_counts,
    update_event_stage,
)


class PipelineRetentionTest(unittest.TestCase):
    def test_build_event_records_groups_consecutive_frames_by_hand(self):
        samples = [
            {"frame": 10, "hand": "left", "contact_points": 8},
            {"frame": 11, "hand": "left", "contact_points": 7},
            {"frame": 12, "hand": "right", "contact_points": 6},
            {"frame": 14, "hand": "left", "contact_points": 5},
        ]

        events = build_event_records(samples)

        self.assertEqual(
            [(event["event_id"], event["hand"], event["start_frame"], event["end_frame"], event["frames"]) for event in events],
            [
                (0, "left", 10, 11, [10, 11]),
                (1, "right", 12, 12, [12]),
                (2, "left", 14, 14, [14]),
            ],
        )
        self.assertEqual(events[0]["representative_frame"], 10)
        self.assertEqual(events[0]["status"], "pending")
        self.assertIsNone(events[0]["failed_stage"])

    def test_update_event_stage_keeps_first_failure_and_stage_statuses(self):
        event = build_event_records([{"frame": 20, "hand": "left"}])[0]

        update_event_stage(event, StageResult("contact_points", True, {"count": 5}))
        update_event_stage(event, StageResult("homography", False, {"reason": "no_ref"}))
        update_event_stage(event, StageResult("heatmap", True, {"path": "ignored.png"}))

        self.assertEqual(event["status"], "discard")
        self.assertEqual(event["failed_stage"], "homography")
        self.assertEqual(event["discard_reason"], "no_ref")
        self.assertIs(event["stages"]["contact_points"]["passed"], True)
        self.assertIs(event["stages"]["homography"]["passed"], False)
        self.assertIs(event["stages"]["heatmap"]["passed"], True)

    def test_summarize_stage_counts_stops_counting_after_failure(self):
        keep = build_event_records([{"frame": 1, "hand": "left"}])[0]
        fail_contact = build_event_records([{"frame": 2, "hand": "left"}])[0]
        fail_homography = build_event_records([{"frame": 3, "hand": "left"}])[0]

        for stage in ["contact_points", "problem3", "heatmap"]:
            update_event_stage(keep, StageResult(stage, True, {}))
        update_event_stage(fail_contact, StageResult("contact_points", False, {"reason": "too_few_points"}))
        update_event_stage(fail_homography, StageResult("contact_points", True, {}))
        update_event_stage(fail_homography, StageResult("problem3", False, {"reason": "no_strict_humanless_frame"}))

        summary = summarize_stage_counts([keep, fail_contact, fail_homography], ["contact_points", "problem3", "heatmap"])

        self.assertEqual(summary, [
            {"stage": "contact_points", "input_events": 3, "kept_events": 2, "discarded_events": 1},
            {"stage": "problem3", "input_events": 2, "kept_events": 1, "discarded_events": 1},
            {"stage": "heatmap", "input_events": 1, "kept_events": 1, "discarded_events": 0},
        ])


if __name__ == "__main__":
    unittest.main()
