import unittest

from scripts.analyze_k50 import adjustment_decision, format_aware_match


class K50AnalysisTests(unittest.TestCase):
    def test_format_aware_grader_accepts_verbose_direct_answers(self):
        self.assertTrue(
            format_aware_match(
                "main-primary-knowledge-scheme",
                "RSA with public exponent 65537",
                ("RSA",),
            )
        )
        self.assertTrue(
            format_aware_match(
                "main-diffecient-admin-criteria",
                "32 chars, lowercase, uppercase, digit, special character",
                ("32 chars, 1 lowercase, 1 uppercase, 1 digit, 1 special char",),
            )
        )

    def test_format_aware_grader_does_not_confuse_strict_inequality(self):
        aliases = ("==", "loose equality", "loose comparison")
        self.assertTrue(format_aware_match("main-motp-operator", "==", aliases))
        self.assertFalse(format_aware_match("main-motp-operator", "!==", aliases))

    def test_predeclared_adjustment_rule(self):
        self.assertEqual(adjustment_decision([1, 45, None])["decision"], "increase")
        self.assertEqual(adjustment_decision([1, 30, None])["decision"], "maintain")
        decreased = adjustment_decision([1, 8, None])
        self.assertEqual(decreased["decision"], "decrease")
        self.assertEqual(decreased["recommended_k"], 8)


if __name__ == "__main__":
    unittest.main()
