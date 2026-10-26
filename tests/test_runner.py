import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_experiment import bounded_tool_record, load_items, parse_seed_list, parse_task_ids


class RunnerBoundaryTests(unittest.TestCase):
    def test_seed_list_is_ordered_unique_and_positive(self):
        self.assertEqual(parse_seed_list("1101, 2202,3303"), (1101, 2202, 3303))
        for invalid in ("", "0,1101", "1101,1101", "not-a-seed"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(Exception):
                    parse_seed_list(invalid)

    def test_tool_output_records_original_length_and_truncation(self):
        result = bounded_tool_record(
            {"ok": True, "tool": "strings", "path": "binary", "output": "x" * 10},
            max_chars=4,
        )
        self.assertTrue(result["output_truncated"])
        self.assertEqual(result["original_output_chars"], 10)
        self.assertTrue(result["output"].startswith("xxxx\n...[truncated"))

    def test_tool_output_can_be_omitted_after_cumulative_budget(self):
        result = bounded_tool_record(
            {"ok": True, "tool": "strings", "path": "binary", "output": "payload"},
            max_chars=0,
        )
        self.assertTrue(result["output_truncated"])
        self.assertEqual(result["original_output_chars"], 7)
        self.assertIn("budget exhausted", result["output"])

    def test_task_id_filter_preserves_prepared_order_and_rejects_missing(self):
        payload = {
            "items": [
                {"id": "a", "split": "main"},
                {"id": "b", "split": "pilot"},
                {"id": "c", "split": "main"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            selected = load_items(path, "main", None, parse_task_ids("c,a"))
            self.assertEqual([item["id"] for item in selected], ["a", "c"])
            with self.assertRaises(ValueError):
                load_items(path, "main", None, ("missing",))
