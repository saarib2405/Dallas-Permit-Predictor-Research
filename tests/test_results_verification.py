"""
Verification tests for the archived result files.

These tests assert the internal consistency of the generated results rather than
the wording of any one manuscript draft. They are intended to survive revisions
to the paper: every assertion here is a property of the pipeline output.

Run from the repository root:

    python -m pytest tests/
"""

import json
import unittest
from pathlib import Path
from statistics import mean, stdev


class TestResultFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        out = cls.root / "outputs"
        cls.pipeline = json.loads((out / "pipeline_results.json").read_text(encoding="utf-8"))
        cls.baselines = json.loads((out / "baseline_comparison_results.json").read_text(encoding="utf-8"))
        cls.shift = json.loads((out / "stage1_shift_eval_results.json").read_text(encoding="utf-8"))
        cls.ablation = json.loads((out / "ablation_results.json").read_text(encoding="utf-8"))
        cls.folds = cls.pipeline["folds"]
        cls.holdout = cls.pipeline["holdout_2020"]

    # ---- structure -------------------------------------------------------

    def test_four_forward_chaining_folds_and_one_holdout(self):
        self.assertEqual(len(self.folds), 4)
        self.assertIn("holdout_2020", self.pipeline)
        self.assertEqual(self.holdout["test_size"], 27457)

    def test_corpus_partition_sums_to_cleaned_total(self):
        development = 99380
        holdout = self.holdout["test_size"]
        self.assertEqual(development + holdout, 126837)

    # ---- burn-in ---------------------------------------------------------

    def test_burn_in_excluded_consistently_in_every_fold(self):
        """Every fold shares the same 2018-01-02 start, so the 60-day
        cold-start exclusion must remove an identical number of records."""
        for f in self.folds:
            self.assertEqual(f["burn_in_excluded"], 6982, f"{f['fold']}")
            self.assertEqual(
                f["train_size_full"] - f["burn_in_excluded"],
                f["train_size"],
                f"{f['fold']}: effective fitting rows do not reconcile",
            )

    def test_holdout_model_applies_the_same_exclusion(self):
        self.assertEqual(self.holdout["burn_in_excluded"], 6982)
        self.assertEqual(
            self.holdout["train_size_full"] - self.holdout["burn_in_excluded"],
            self.holdout["train_size"],
        )
        self.assertEqual(self.holdout["train_size"], 92398)

    def test_training_windows_expand_monotonically(self):
        sizes = [f["train_size"] for f in self.folds]
        self.assertEqual(sizes, sorted(sizes))

    # ---- reported aggregates --------------------------------------------

    def test_cross_fold_log_r2_mean_and_sample_sd(self):
        values = [f["r2_log"] for f in self.folds]
        self.assertAlmostEqual(mean(values), 0.7145, places=4)
        self.assertAlmostEqual(stdev(values), 0.0416, places=4)  # ddof=1

    def test_cross_fold_dollar_r2_mean(self):
        values = [f["r2_hard"] for f in self.folds]
        self.assertAlmostEqual(mean(values), 0.4542, places=4)

    def test_holdout_headline_values(self):
        self.assertAlmostEqual(self.holdout["r2_log"], 0.6349, places=4)
        self.assertAlmostEqual(self.holdout["r2_hard"], 0.5192, places=4)
        self.assertAlmostEqual(self.holdout["s1_auc"], 0.9678, places=4)

    def test_duan_factors_are_well_behaved_for_the_hurdle_model(self):
        """The Stage 2 smearing factor should stay in a narrow band. A value
        far outside it indicates the estimator is outlier-dominated."""
        for f in self.folds + [self.holdout]:
            self.assertTrue(
                1.4 < f["duan_factor"] < 1.7,
                f"{f.get('fold')}: Duan factor {f['duan_factor']} outside expected band",
            )

    # ---- comparators -----------------------------------------------------

    def test_ridge_smearing_factor_is_outlier_dominated(self):
        """Reported as a finding: the single-stage log-linear comparator
        produces a smearing factor an order of magnitude above the hurdle's."""
        factors = [f["log_linear_ridge"]["duan_factor"] for f in self.baselines["folds"]]
        self.assertTrue(min(factors) > 20, f"min Duan factor {min(factors)}")
        self.assertTrue(max(factors) < 50, f"max Duan factor {max(factors)}")

    def test_hurdle_beats_tweedie_on_aggregate_dollar_r2(self):
        hurdle = mean([f["r2_hard"] for f in self.folds])
        tweedie = mean([f["tweedie_xgboost"]["r2_dollars"] for f in self.baselines["folds"]])
        self.assertGreater(hurdle, tweedie)
        self.assertAlmostEqual(tweedie, 0.3888, places=4)

    def test_tweedie_wins_some_folds_on_dollar_metrics(self):
        """Guards the trade-off claim: the comparison is not a clean sweep,
        and the manuscript must not describe it as one."""
        tweedie = self.baselines["folds"]
        wins_mae = [
            i for i, (t, h) in enumerate(zip(tweedie, self.folds))
            if t["tweedie_xgboost"]["mae_dollars"] < h["mae_hard"]
        ]
        self.assertTrue(wins_mae, "expected Tweedie to win MAE in at least one fold")
        self.assertIn(1, wins_mae, "expected a Tweedie MAE win in Fold 2")
        self.assertIn(2, wins_mae, "expected a Tweedie MAE win in Fold 3")

    # ---- recording-pattern shift ----------------------------------------

    def test_building_permit_zero_rate_in_2020(self):
        b = self.shift["building_permits_breakdown"]
        self.assertEqual(b["building_permits_total_2020"], 8997)
        self.assertEqual(b["building_permits_zero_count_2020"], 4095)
        self.assertAlmostEqual(b["building_permits_zero_rate_2020"], 4095 / 8997, places=6)
        self.assertAlmostEqual(round(b["building_permits_zero_rate_2020"] * 100, 2), 45.52, places=2)

    def test_pre_shift_classifier_fails_on_2020(self):
        """A classifier trained only on pre-shift data should score at or
        below chance on the holdout."""
        pre = self.shift["metrics_preshift_on_2020"]
        full = self.shift["metrics_full_on_2020"]
        self.assertLess(pre["roc_auc"], 0.5)
        self.assertGreater(full["roc_auc"], 0.95)
        self.assertLess(pre["recall"], full["recall"])

    def test_pre_shift_building_recall_collapse(self):
        b = self.shift["building_permits_breakdown"]
        self.assertLess(b["preshift_trained_bldg_recall"], 0.05)
        self.assertGreater(b["full_trained_bldg_recall"], 0.90)

    # ---- ablation --------------------------------------------------------

    def test_parser_fix_dollar_effect_is_mixed(self):
        """Guards against describing the parser fix as a uniform improvement."""
        v2 = {"Fold 1": 0.5124, "Fold 2": 0.2007, "Fold 3": 0.4218, "Fold 4": 0.5401}
        deltas = [f["r2_hard"] - v2[f["fold"]] for f in self.folds]
        self.assertTrue(any(d > 0 for d in deltas), "expected at least one improvement")
        self.assertTrue(any(d < 0 for d in deltas), "expected at least one regression")

    def test_fold1_dollar_r2_delta_arithmetic(self):
        f1 = next(f for f in self.folds if f["fold"] == "Fold 1")
        self.assertAlmostEqual(f1["r2_hard"] - 0.5124, -0.0056, places=4)

    # ---- feature importance ---------------------------------------------

    def test_history_features_contribute_little_after_correction(self):
        """Central finding: once leakage is removed the engineered history
        features carry under one percent of Stage 2 gain importance."""
        import csv
        path = self.root / "outputs" / "feature_importance_stage2.csv"
        rows = list(csv.reader(path.open(encoding="utf-8")))[1:]
        history = {
            "contractor_permit_count",
            "contractor_city_freq",
            "permit_code_freq",
            "land_use_freq",
            "is_repeat_contractor",
        }
        total = sum(float(r[1]) for r in rows)
        share = sum(float(r[1]) for r in rows if r[0] in history) / total
        self.assertLess(share, 0.01, f"history share {share:.4%} exceeds 1%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
