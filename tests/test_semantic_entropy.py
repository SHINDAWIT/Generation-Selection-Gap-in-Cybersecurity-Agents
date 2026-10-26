import json
import math
from pathlib import Path
import unittest

from cisc_rollout.normalize import normalize_answer
from scripts.analyze_semantic_entropy import (
    binary_auroc,
    classify_semantic_answer,
    grouped_risk_coverage,
    normalized_entropy,
    shannon_entropy,
)


class UncertaintyMetricTests(unittest.TestCase):
    def test_entropy_known_distribution(self):
        self.assertAlmostEqual(shannon_entropy([49, 1]), 0.0980391133, places=9)
        self.assertEqual(shannon_entropy([50]), 0.0)
        self.assertAlmostEqual(normalized_entropy([1, 1]), 1.0)

    def test_auroc_counts_ties_as_half(self):
        self.assertEqual(binary_auroc([True, False], [1.0, 0.0]), 1.0)
        self.assertEqual(binary_auroc([True, False], [0.0, 0.0]), 0.5)

    def test_grouped_risk_coverage_keeps_ties_together(self):
        points, aurc = grouped_risk_coverage(
            [True, False, True], [0.0, 0.0, 1.0]
        )
        self.assertEqual(points[0]["accepted"], 2)
        self.assertAlmostEqual(points[0]["risk"], 0.5)
        self.assertAlmostEqual(points[-1]["coverage"], 1.0)
        self.assertGreater(aurc, 0.0)


class SemanticClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotations = json.loads(
            Path("data/annotations/k50_semantic_clusters.json").read_text(
                encoding="utf-8"
            )
        )

    def classify(self, task_id, answer):
        return classify_semantic_answer(
            task_id, normalize_answer(answer), self.annotations
        )

    def test_security_significant_operators_are_distinct(self):
        correct = self.classify("main-motp-operator", "==")
        wrong = self.classify("main-motp-operator", "!==")
        self.assertEqual(correct["cluster_id"], "loose_equality")
        self.assertEqual(correct["correctness"], "correct")
        self.assertEqual(wrong["cluster_id"], "strict_inequality")
        self.assertEqual(wrong["correctness"], "incorrect")

    def test_natural_language_rsa_variants_collapse(self):
        short = self.classify("main-primary-knowledge-scheme", "RSA")
        verbose = self.classify(
            "main-primary-knowledge-scheme",
            "RSA encryption with public exponent e=65537 and modulus n",
        )
        self.assertEqual(short["cluster_id"], verbose["cluster_id"])

    def test_complete_and_incomplete_admin_criteria_are_distinct(self):
        complete = self.classify(
            "main-diffecient-admin-criteria",
            "32 chars, lowercase, uppercase, digit, special char",
        )
        incomplete = self.classify(
            "main-diffecient-admin-criteria",
            "chars, lowercase, uppercase, digit, special char",
        )
        self.assertEqual(complete["correctness"], "correct")
        self.assertEqual(incomplete["correctness"], "incorrect")


if __name__ == "__main__":
    unittest.main()
