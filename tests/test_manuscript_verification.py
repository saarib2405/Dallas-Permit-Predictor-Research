import json
import re
import unittest
from pathlib import Path


class TestManuscriptVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.paper_path = cls.repo_root / "docs" / "research_paper_draft.md"
        cls.fold2_path = cls.repo_root / "docs" / "fold2_and_parser_findings.md"
        cls.shift_eval_path = cls.repo_root / "outputs" / "stage1_shift_eval_results.json"

        cls.assertTrue(cls.paper_path.exists(), f"Missing {cls.paper_path}")
        cls.paper_text = cls.paper_path.read_text(encoding="utf-8")

        if cls.fold2_path.exists():
            cls.fold2_text = cls.fold2_path.read_text(encoding="utf-8")
        else:
            cls.fold2_text = ""

        if cls.shift_eval_path.exists():
            with open(cls.shift_eval_path, "r", encoding="utf-8") as f:
                cls.shift_eval_data = json.load(f)
        else:
            cls.shift_eval_data = {}

    def test_task1_table5_2020_row_zero_value_permits(self):
        """
        Task 1: Table 5, 2020 row must say 4,095 (45.52%), matching cleaned data
        and stage1_shift_eval_results.json, and not 4,073 (45.27%).
        """
        # 1. Verify against stage1_shift_eval_results.json
        self.assertIn("building_permits_breakdown", self.shift_eval_data)
        bldg_breakdown = self.shift_eval_data["building_permits_breakdown"]
        self.assertEqual(bldg_breakdown["building_permits_total_2020"], 8997)
        self.assertEqual(bldg_breakdown["building_permits_zero_count_2020"], 4095)
        rate = bldg_breakdown["building_permits_zero_rate_2020"]
        self.assertAlmostEqual(rate, 4095 / 8997, places=5)
        self.assertAlmostEqual(round(rate * 100, 2), 45.52, places=2)

        # 2. Verify Table 5 in docs/research_paper_draft.md
        # Locate Table 5 2020 row: | **2020** | Holdout Period | 8,997 | 4,095 | **45.52%** |
        table5_match = re.search(
            r"\|\s*\*\*2020\*\*\s*\|\s*Holdout Period\s*\|\s*8,997\s*\|\s*([^|]+)\|\s*([^|]+)\|",
            self.paper_text,
        )
        self.assertIsNotNone(table5_match, "Could not find Table 5 2020 row in research_paper_draft.md")
        zero_count_col = table5_match.group(1).strip()
        zero_rate_col = table5_match.group(2).strip()

        self.assertEqual(zero_count_col, "4,095", f"Expected '4,095' in Table 5 count col, got '{zero_count_col}'")
        self.assertIn("45.52%", zero_rate_col, f"Expected '45.52%' in Table 5 rate col, got '{zero_rate_col}'")

        # Ensure outdated numbers are completely purged
        self.assertNotIn("4,073", self.paper_text, "Outdated count '4,073' found in paper draft")
        self.assertNotIn("45.27%", self.paper_text, "Outdated rate '45.27%' found in paper draft")

        # 3. Check docs/fold2_and_parser_findings.md if present
        if self.fold2_text:
            self.assertIn("4,095", self.fold2_text)
            self.assertIn("45.52%", self.fold2_text)
            self.assertNotIn("4,073", self.fold2_text)
            self.assertNotIn("45.27%", self.fold2_text)

    def test_task2_table9_fold1_dollar_r2_hard_delta(self):
        """
        Task 2: Table 9, Fold 1, Dollar R² (Hard), 'v3.1 vs. v2' column must say -0.0056,
        matching 0.5124 to 0.5068 and the text in Section 6.1.
        """
        # Locate Table 9 Fold 1 Dollar R2 row:
        # | | Dollar $R^2$ (Hard) | 0.5124 | **0.5184** | 0.5068 | $+0.0060$ | $-0.0056$ |
        row_match = re.search(
            r"\|\s*\|\s*Dollar\s*\$R\^2\$\s*\(Hard\)\s*\|\s*0\.5124\s*\|\s*\*\*0\.5184\*\*\s*\|\s*0\.5068\s*\|\s*\$\+0\.0060\$\s*\|\s*([^|]+)\|",
            self.paper_text,
        )
        self.assertIsNotNone(row_match, "Could not find Fold 1 Dollar R^2 (Hard) row in Table 9")
        delta_col = row_match.group(1).strip()
        self.assertIn("-0.0056", delta_col, f"Expected '-0.0056' in delta column, got '{delta_col}'")

        # Ensure -0.0560 is not present
        self.assertNotIn("-0.0560", self.paper_text, "Outdated typo '-0.0560' found in paper draft")

        # Verify arithmetic consistency: 0.5068 - 0.5124 = -0.0056
        self.assertAlmostEqual(0.5068 - 0.5124, -0.0056, places=4)

        # Verify Section 6.1 text mentions ($0.5124 \to 0.5068$, $-0.0056$)
        self.assertIn("-0.0056", self.paper_text)
        self.assertIn("0.5124 \\to 0.5068", self.paper_text)

    def test_task3_section_6_1_point3_stage1_metrics_wording(self):
        """
        Task 3: Section 6.1 point 3 must state that recall and F1 improve in every fold,
        and remove claims that all Stage 1 metrics improve consistently (since AUC dips slightly).
        """
        self.assertNotIn(
            "Stage 1 classification metrics and zero-leakage suppression improve consistently",
            self.paper_text,
            "Old over-strong claim found in Section 6.1",
        )
        self.assertNotIn(
            "improve consistently",
            self.paper_text,
            "'improve consistently' found in paper draft",
        )

        # Check for the refined wording
        self.assertIn(
            "recall and F1 improve in every fold",
            self.paper_text,
            "Missing 'recall and F1 improve in every fold' in Section 6.1",
        )

    def test_task4_section_4_3_takeaway_softened_wording(self):
        """
        Task 4: Section 4.3 takeaway must use softer wording (e.g., 'is consistent with')
        instead of 'demonstrates' and 'establishing that prospective generalization reflects empirical adaptation'.
        """
        self.assertNotIn(
            "The counterfactual experiment demonstrates that",
            self.paper_text,
            "'demonstrates that' still present in Section 4.3 takeaway",
        )
        self.assertNotIn(
            "establishing that prospective generalization reflects empirical adaptation",
            self.paper_text,
            "'establishing that prospective generalization reflects empirical adaptation' still present",
        )

        # Verify softer wording
        self.assertIn(
            "The counterfactual findings are consistent with the hypothesis that",
            self.paper_text,
            "Missing softened phrase 'consistent with the hypothesis that' in Section 4.3 takeaway",
        )
        self.assertIn(
            "which is consistent with prospective generalization reflecting empirical adaptation",
            self.paper_text,
            "Missing softened phrase 'which is consistent with prospective generalization reflecting' in Section 4.3",
        )

    def test_optional_title_and_table1b_wording(self):
        """
        Optional task: Title uses 'Values' and Table 1B row uses 'Positive-value permit filings'.
        """
        # Title check
        self.assertIn(
            "# Predictive Modeling of Municipal Building Permit Values:",
            self.paper_text,
            "Paper title should use 'Values' instead of 'Valuations'",
        )

        # Table 1B row check
        self.assertIn(
            "Positive-value permit filings subject to Stage 2 log regression",
            self.paper_text,
            "Table 1B should use 'Positive-value permit filings' instead of 'Construction projects'",
        )
        self.assertNotIn(
            "Construction projects subject to Stage 2 log regression",
            self.paper_text,
            "Old 'Construction projects' wording found in Table 1B",
        )

        # Maximum Declared Value check
        self.assertIn(
            "Maximum Declared Value",
            self.paper_text,
            "Table 1B should use 'Maximum Declared Value'",
        )


if __name__ == "__main__":
    unittest.main()
