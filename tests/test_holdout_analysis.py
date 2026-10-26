import unittest

from scripts.analyze_holdout import SEEDS_20, analyze_task, build_analysis


def record(task_id, seed, answer=None, parse_failed=False):
    return {
        "task_id": task_id,
        "seed": seed,
        "split": "holdout",
        "final": None if answer is None else {"final": answer, "evidence": []},
        "parse_failed": parse_failed,
        "generated_tokens": 1,
        "wall_seconds": 0.1,
    }


class HoldoutAnalysisTests(unittest.TestCase):
    def test_support_threshold_abstains_low_support_modal(self):
        item = {
            "id": "holdout-a",
            "source_task_path": "task/a",
            "gold_aliases": ["right"],
        }
        rows = [
            record("holdout-a", seed, "right" if index < 11 else f"wrong-{index}")
            for index, seed in enumerate(SEEDS_20)
        ]
        task = analyze_task(item, rows)
        self.assertTrue(task["oracle_correct"])
        self.assertTrue(task["abstained"])
        self.assertIsNone(task["selected_correct"])

    def test_majority_and_oracle_are_separate(self):
        prepared = {
            "items": [
                {
                    "id": "holdout-a",
                    "source_task_path": "task/a",
                    "gold_aliases": ["right"],
                }
            ]
        }
        rows = [
            record("holdout-a", seed, "wrong" if index < 12 else "right")
            for index, seed in enumerate(SEEDS_20)
        ]
        analysis = build_analysis(prepared, rows)
        majority = analysis["selectors"]["exact_majority_support_ge_0_60"]
        oracle = analysis["selectors"]["exact_oracle_pass_at_20"]
        self.assertEqual(majority["answered"], 1)
        self.assertEqual(majority["correct"], 0)
        self.assertEqual(majority["incorrect"], 1)
        modal = analysis["selectors"]["exact_modal_without_support_abstention"]
        self.assertEqual(modal["answered"], 1)
        self.assertEqual(modal["correct"], 0)
        self.assertEqual(oracle["correct"], 1)
        self.assertIn("bootstrap_95_ci", analysis)
        self.assertIsNotNone(
            analysis["bootstrap_95_ci"]["selected_overall_accuracy"]["lower"]
        )
        self.assertIsNotNone(
            analysis["bootstrap_95_ci"]["exact_modal_accuracy"]["lower"]
        )
        self.assertIsNotNone(analysis["bootstrap_95_ci"]["selective_risk"]["lower"])
        self.assertEqual(analysis["mean_generated_tokens_per_rollout"], 1.0)
        self.assertEqual(analysis["mean_wall_seconds_per_rollout"], 0.1)
        headroom = analysis["selectors"]["reachable_headroom_recovery"]
        self.assertEqual(headroom["reachable_headroom_tasks"], 1)
        self.assertEqual(headroom["recovered_fraction"], 0.0)
        abstention = analysis["selectors"]["abstention_effect"]
        self.assertEqual(abstention["wrong_modal_intercepted"], 0)
        self.assertEqual(abstention["correct_modal_abstained"], 0)
        self.assertEqual(abstention["risk_reduction"], 0.0)

    def test_selective_risk_ci_is_undefined_when_policy_never_answers(self):
        prepared = {
            "items": [
                {
                    "id": "holdout-a",
                    "source_task_path": "task/a",
                    "gold_aliases": ["right"],
                }
            ]
        }
        rows = [record("holdout-a", seed, f"answer-{index}") for index, seed in enumerate(SEEDS_20)]
        analysis = build_analysis(prepared, rows)
        majority = analysis["selectors"]["exact_majority_support_ge_0_60"]
        self.assertEqual(majority["answered"], 0)
        self.assertIsNone(majority["selective_risk"])
        self.assertIsNone(analysis["bootstrap_95_ci"]["selective_risk"]["lower"])

    def test_incomplete_grid_fails(self):
        prepared = {
            "items": [
                {
                    "id": "holdout-a",
                    "source_task_path": "task/a",
                    "gold_aliases": ["right"],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_analysis(prepared, [record("holdout-a", SEEDS_20[0], "right")])


if __name__ == "__main__":
    unittest.main()
