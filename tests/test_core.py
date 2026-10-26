from pathlib import Path
import tempfile
import unittest

from src.cisc_rollout.metrics import bootstrap_percentile_ci, evaluate_tasks, paired_delta_stat
from src.cisc_rollout.normalize import answers_match, normalize_answer
from src.cisc_rollout.schema import Evidence, FinalAnswer, Rollout, TaskItem, ToolObservation, parse_model_output, parse_turn
from src.cisc_rollout.schema import rollout_from_record
from src.cisc_rollout.selectors import (
    agreement_abstention,
    normalized_majority,
    oracle_pass_at_k,
    single_first,
    tool_grounded_majority,
)
from src.cisc_rollout.tools import ReadOnlyToolset, ToolError, ToolSandbox


def task(task_id="t1", aliases=("secret.py",)):
    return TaskItem(task_id, "main", "example/task", "question?", aliases)


def rollout(seed, answer, *, quote="secret.py", path="a.txt", parse_failed=False, tool_failed=False):
    final = None if answer is None else FinalAnswer(answer, (Evidence(path, quote),))
    observations = (ToolObservation("read_file", path, f"contains {quote} here"),)
    return Rollout("t1", seed, final, observations, parse_failed=parse_failed, tool_failed=tool_failed)


class NormalizeTests(unittest.TestCase):
    def test_normalizes_case_markdown_quotes_and_spacing(self):
        self.assertEqual(normalize_answer(' **"Secret.py"** \n'), "secret.py")
        self.assertTrue(answers_match("`BASE64`", ("base64",)))

    def test_preserves_security_significant_boundary_punctuation(self):
        self.assertEqual(normalize_answer("!=="), "!==")
        self.assertEqual(normalize_answer(".git"), ".git")
        self.assertEqual(normalize_answer("__main__"), "__main__")
        self.assertFalse(answers_match("!==", ("==",)))


class ParseTests(unittest.TestCase):
    def test_parses_embedded_final_json(self):
        parsed = parse_turn('text before {"final":"secret.py","evidence":[{"path":"a.txt","quote":"secret.py"}]}')
        self.assertEqual(parsed["final"], "secret.py")
        self.assertEqual(parse_model_output('{"action":"read_file","path":"a.txt"}')["action"], "read_file")

    def test_rejects_bad_json_turn(self):
        with self.assertRaises(ValueError):
            parse_turn("not json")

    def test_rollout_from_record_accepts_jsonl_shape(self):
        parsed = rollout_from_record(
            {
                "task_id": "t1",
                "seed": 1101,
                "final": "secret.py",
                "evidence": [{"path": "a.txt", "quote": "secret.py"}],
                "tool_results": [{"tool": "read_file", "path": "a.txt", "output": "secret.py", "ok": True}],
            }
        )
        self.assertTrue(parsed.valid)
        self.assertTrue(parsed.is_tool_grounded)


class SelectorTests(unittest.TestCase):
    def test_majority_correct_and_single_wrong(self):
        rollouts = [
            rollout(1101, "wrong"),
            rollout(2202, "Secret.py"),
            rollout(3303, "secret.py"),
            rollout(4404, "other"),
            rollout(5505, "secret.py"),
        ]
        self.assertFalse(single_first(rollouts, ("secret.py",)).correct)
        selected = normalized_majority(rollouts, ("secret.py",))
        self.assertEqual(selected.normalized_answer, "secret.py")
        self.assertEqual(selected.support, 3)
        self.assertTrue(selected.correct)

    def test_wrong_consensus_is_visible_in_metrics(self):
        items = [task()]
        rollouts_by_task = {"t1": [rollout(seed, "wrong") for seed in (1101, 2202, 3303)] + [rollout(4404, "secret.py")]}
        summary = evaluate_tasks(items, rollouts_by_task)
        self.assertEqual(summary["secondary"]["wrong_consensus_rate"], 1.0)
        self.assertEqual(summary["selectors"]["majority"]["overall_accuracy"], 0.0)
        self.assertEqual(summary["selectors"]["oracle"]["overall_accuracy"], 1.0)

    def test_agreement_abstains_below_three_of_five(self):
        rollouts = [
            rollout(1101, "a"),
            rollout(2202, "a"),
            rollout(3303, "b"),
            rollout(4404, "c"),
            rollout(5505, "d"),
        ]
        selected = agreement_abstention(rollouts, ("a",))
        self.assertTrue(selected.abstained)
        self.assertEqual(selected.support, 2)

    def test_evaluation_accepts_scaled_agreement_threshold(self):
        items = [task()]
        rollouts_by_task = {
            "t1": [rollout(seed, "secret.py") for seed in range(1, 15)]
            + [rollout(15, "other")]
        }
        below = evaluate_tasks(items, rollouts_by_task, agreement_threshold=15)
        self.assertEqual(
            below["selectors"]["agreement_abstention"]["coverage"], 0.0
        )
        at_threshold = evaluate_tasks(
            items,
            {"t1": [rollout(seed, "secret.py") for seed in range(1, 16)]},
            agreement_threshold=15,
        )
        self.assertEqual(
            at_threshold["selectors"]["agreement_abstention"]["coverage"], 1.0
        )

    def test_oracle_only_success(self):
        rollouts = [
            rollout(1101, "wrong"),
            rollout(2202, "also wrong"),
            rollout(3303, "secret.py"),
        ]
        self.assertFalse(normalized_majority(rollouts, ("secret.py",)).correct)
        self.assertTrue(oracle_pass_at_k(rollouts, ("secret.py",)).correct)

    def test_tool_grounded_filters_unsupported_answers(self):
        unsupported = Rollout("t1", 1101, FinalAnswer("secret.py", (Evidence("a.txt", "secret.py"),)), ())
        supported = rollout(2202, "wrong", quote="wrong")
        selected = tool_grounded_majority([unsupported, supported], ("secret.py",))
        self.assertEqual(selected.answer, "wrong")
        self.assertFalse(selected.correct)

    def test_parse_and_tool_failures_are_counted(self):
        items = [task()]
        rollouts_by_task = {"t1": [rollout(1101, None, parse_failed=True), rollout(2202, "secret.py", tool_failed=True)]}
        summary = evaluate_tasks(items, rollouts_by_task)
        self.assertEqual(summary["secondary"]["parse_failure_rate"], 0.5)
        self.assertEqual(summary["secondary"]["tool_failure_rate"], 0.5)


class MetricsTests(unittest.TestCase):
    def test_bootstrap_is_seeded_and_task_level(self):
        tasks = [task("t1"), task("t2")]
        rollouts_by_task = {
            "t1": [rollout(1101, "secret.py")],
            "t2": [Rollout("t2", 1101, FinalAnswer("wrong", ()), ())],
        }
        first = bootstrap_percentile_ci(tasks, rollouts_by_task, paired_delta_stat("oracle"), samples=50, seed=7)
        second = bootstrap_percentile_ci(tasks, rollouts_by_task, paired_delta_stat("oracle"), samples=50, seed=7)
        self.assertEqual(first, second)


class ToolTests(unittest.TestCase):
    def test_read_only_tools_are_path_confined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha\nsecret.py\n", encoding="utf-8")
            tools = ReadOnlyToolset(root)
            self.assertIn("a.txt", tools.list_files().output)
            self.assertIn("secret.py", tools.read_file("a.txt").output)
            self.assertIn("secret.py", tools.strings("a.txt").output)
            self.assertIn("00000000", tools.hexdump("a.txt").output)
            executed = ToolSandbox(root).execute({"action": "read_file", "path": "a.txt"})
            self.assertTrue(executed["ok"])
            self.assertIn("secret.py", executed["output"])
            with self.assertRaises(ToolError):
                tools.read_file("../outside.txt")
            with self.assertRaises(ToolError):
                tools.read_file(str(root / "a.txt"))

    def test_tools_resolve_unique_basenames_and_bound_binary_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "release" / "challenge"
            nested.mkdir(parents=True)
            (nested / "source.py").write_text("from secret import FLAG\n", encoding="utf-8")
            (nested / "program").write_bytes(b"\x7fELF\x00binary")
            tools = ReadOnlyToolset(root)
            resolved = tools.read_file("source.py")
            self.assertEqual(resolved.path, "release/challenge/source.py")
            binary = tools.execute({"action": "read_file", "path": "program"})
            self.assertTrue(binary["ok"])
            self.assertIn("BINARY FILE: 11 bytes", binary["output"])
            self.assertNotIn("\x00", binary["output"])

    def test_tools_resolve_identical_duplicates_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "short").mkdir()
            (root / "long" / "nested").mkdir(parents=True)
            (root / "short" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
            (root / "long" / "nested" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
            resolved = ReadOnlyToolset(root).read_file("main.rs")
            self.assertEqual(resolved.path, "short/main.rs")

    def test_tools_report_ambiguous_different_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "main.rs").write_text("first\n", encoding="utf-8")
            (root / "b" / "main.rs").write_text("second\n", encoding="utf-8")
            result = ReadOnlyToolset(root).execute({"action": "read_file", "path": "main.rs"})
            self.assertFalse(result["ok"])
            self.assertIn("a/main.rs", result["error"])
            self.assertIn("b/main.rs", result["error"])

    def test_basename_fallback_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "sandbox"
            nested = root / "nested"
            nested.mkdir(parents=True)
            outside = parent / "outside.txt"
            outside.write_text("host secret\n", encoding="utf-8")
            (nested / "leak.txt").symlink_to(outside)
            result = ReadOnlyToolset(root).execute(
                {"action": "read_file", "path": "leak.txt"}
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["output"], "")
            self.assertNotIn("host secret", result["error"])


if __name__ == "__main__":
    unittest.main()
