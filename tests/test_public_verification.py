"""Publication checks: reject missing evidence and preserve decision boundaries."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_public_results import analyze, task_result, validate_ledger, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def record(seed, answer, correct=False):
    return {"seed": seed, "answer_key": answer, "strict_correct": correct}


class DecisionBoundaryTests(unittest.TestCase):
    def test_development_tie_uses_first_seed_and_k20_tie_abstains(self):
        rows = [record(1101, "z", True), record(2202, "a")]
        self.assertTrue(task_result(rows, tie_abstains=False)["modal_correct"])
        self.assertFalse(task_result(rows, tie_abstains=True)["modal_answered"])

    def test_support_denominator_uses_valid_answers(self):
        rows = [record(i, "wrong" if i < 9 else None) for i in range(20)]
        result = task_result(rows, tie_abstains=True)
        self.assertEqual(result["invalid"], 11)
        self.assertEqual(result["support"], 1.0)
        self.assertTrue(result["policy_answered"])
        self.assertFalse(result["policy_correct"])

    def test_oracle_sees_rare_correct_answer_without_changing_mode(self):
        rows = [record(i, "wrong") for i in range(49)] + [record(49, "right", True)]
        result = task_result(rows, tie_abstains=False)
        self.assertTrue(result["oracle_correct"])
        self.assertFalse(result["modal_correct"])


class EvidenceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohorts = json.loads((ROOT / "data/public/cohorts.json").read_text())
        cls.rows = [json.loads(line) for line in (ROOT / "data/public/rollout_answers.jsonl").read_text().splitlines()]

    def test_complete_ledger_reproduces_reference(self):
        validate_ledger(self.rows, self.cohorts)
        reference = json.loads((ROOT / "results/paper_results.json").read_text())
        self.assertEqual(analyze(self.rows), reference["counts"])

    def test_missing_and_duplicate_seeds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_ledger(self.rows[:-1], self.cohorts)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ledger(self.rows + [self.rows[0]], self.cohorts)

    def test_inconsistent_labels_for_same_answer_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        first = next(row for row in rows if row["answer_key"] is not None)
        same = next(row for row in rows if row is not first and row["task_id"] == first["task_id"]
                    and row["answer_key"] == first["answer_key"])
        same["strict_correct"] = not same["strict_correct"]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_ledger(rows, self.cohorts)

    def test_tampered_or_missing_published_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evidence.txt"
            path.write_bytes(b"preserved")
            manifest = {"files": {"evidence.txt": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                    "bytes": path.stat().st_size}}}
            (root / "publication_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_manifest(root), 1)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_manifest(root)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing file"):
                verify_manifest(root)


if __name__ == "__main__":
    unittest.main()
