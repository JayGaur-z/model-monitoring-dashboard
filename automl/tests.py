import numpy as np
import pandas as pd
from django.test import TestCase, Client
from django.urls import reverse
from automl.models import TrainedModel, PredictionLog
from automl.drift import compute_numerical_drift, compute_categorical_drift, detect_data_drift

class StatisticalDriftUnitTests(TestCase):
    """
    Unit tests for Kolmogorov-Smirnov (Numerical) and Chi-Square (Categorical)
    data drift detection algorithms.
    """

    def setUp(self):
        np.random.seed(42)

    def test_numerical_drift_detected_when_distribution_shifts(self):
        """Verifies KS test flags numerical drift when distribution mean shifts significantly."""
        reference = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=200))
        # Artificially shifted production sample (mean shifted from 0 to 4)
        production = pd.Series(np.random.normal(loc=4.0, scale=1.0, size=50))

        result = compute_numerical_drift(reference, production, alpha=0.05)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['test'], 'Kolmogorov-Smirnov')
        self.assertTrue(result['drift_detected'], "Expected KS test to detect numerical drift")
        self.assertLess(result['p_value'], 0.05)
        self.assertGreater(result['statistic'], 0.5)

    def test_numerical_distribution_stable_when_unchanged(self):
        """Verifies KS test confirms stability when reference and production come from identical distributions."""
        reference = pd.Series(np.random.normal(loc=10.0, scale=2.0, size=300))
        production = pd.Series(np.random.normal(loc=10.0, scale=2.0, size=60))

        result = compute_numerical_drift(reference, production, alpha=0.05)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertFalse(result['drift_detected'], "Expected KS test to report stable distribution")
        self.assertGreaterEqual(result['p_value'], 0.05)

    def test_categorical_drift_detected_when_frequencies_shift(self):
        """Verifies Chi-Square test flags categorical drift when class proportions change significantly."""
        # Baseline: 80% Category A, 20% Category B
        reference = pd.Series(['Category_A'] * 400 + ['Category_B'] * 100)
        # Production: 10% Category A, 90% Category B (severe drift)
        production = pd.Series(['Category_A'] * 10 + ['Category_B'] * 90)

        result = compute_categorical_drift(reference, production, alpha=0.05)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['test'], 'Chi-Square')
        self.assertTrue(result['drift_detected'], "Expected Chi-Square test to detect categorical drift")
        self.assertLess(result['p_value'], 0.05)

    def test_categorical_distribution_stable_when_unchanged(self):
        """Verifies Chi-Square test confirms stability when category proportions are preserved."""
        reference = pd.Series(['Basic'] * 300 + ['Standard'] * 300 + ['Premium'] * 300)
        production = pd.Series(['Basic'] * 30 + ['Standard'] * 30 + ['Premium'] * 30)

        result = compute_categorical_drift(reference, production, alpha=0.05)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertFalse(result['drift_detected'], "Expected Chi-Square to report stable categorical distribution")
        self.assertGreaterEqual(result['p_value'], 0.05)

    def test_missing_data_and_nans_handled_safely(self):
        """Verifies that missing values (NaNs/Nones) are cleaned and do not trigger crashes."""
        ref_with_nans = pd.Series([1.0, 2.0, np.nan, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0])
        prod_with_nans = pd.Series([1.1, np.nan, 2.1, 4.2, None, 6.1, 7.2, 8.1, 9.2, 10.1])

        num_res = compute_numerical_drift(ref_with_nans, prod_with_nans, alpha=0.05)
        self.assertEqual(num_res['status'], 'SUCCESS')
        self.assertIsNotNone(num_res['p_value'])

        cat_ref_nans = pd.Series(['A', None, 'B', np.nan, 'A', 'B', 'A', 'B', 'A', 'B'])
        cat_prod_nans = pd.Series(['A', 'B', np.nan, None, 'A', 'B', 'A', 'B', 'A', 'B'])
        cat_res = compute_categorical_drift(cat_ref_nans, cat_prod_nans, alpha=0.05)
        self.assertEqual(cat_res['status'], 'SUCCESS')

    def test_insufficient_samples_guard_prevents_false_alerts(self):
        """
        Verifies that when production sample size is below the minimum threshold (e.g. < 25),
        detect_data_drift returns INSUFFICIENT_DATA and drift_detected is False.
        """
        model = TrainedModel.objects.create(
            model_name="Small_Sample_Test_Model",
            pkl_file_path="dummy.pkl",
            feature_columns=["age", "income"],
            target_column="target",
        )
        # Create only 5 logs (below threshold of 25)
        for i in range(5):
            PredictionLog.objects.create(
                trained_model=model,
                input_data={"age": 25 + i, "income": 50000 + i * 1000},
                prediction_result="0",
                response_time_ms=20.0
            )

        report = detect_data_drift(model, sample_window=100, alpha=0.05, min_samples=25)
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['drift_detected'])
        self.assertEqual(report['sample_count'], 5)
        self.assertIn("minimum 25 required", report['message'])

    def test_latency_decoupled_from_statistical_drift(self):
        """
        Verifies that high latency (> 500ms) does NOT trigger statistical data drift.
        Confirms separation of DevOps SLA monitoring from MLOps statistical drift.
        """
        model = TrainedModel.objects.create(
            model_name="High_Latency_Model",
            pkl_file_path="dummy.pkl",
            feature_columns=["feature_a"],
            target_column="target",
        )
        # Create 10 logs with extreme latency (1200ms)
        for i in range(10):
            PredictionLog.objects.create(
                trained_model=model,
                input_data={"feature_a": 10.0},
                prediction_result="1",
                response_time_ms=1200.0  # High response time
            )

        # Statistical drift should be False (or INSUFFICIENT_DATA), not True simply because latency is high
        report = detect_data_drift(model, min_samples=25)
        self.assertFalse(report['drift_detected'], "High latency must not trigger statistical data drift alert")

    def test_monitor_view_decouples_latency_and_drift_context(self):
        """Verifies the monitor view passes distinct latency_alert and drift_alert flags."""
        model = TrainedModel.objects.create(
            model_name="Context_Test_Model",
            pkl_file_path="dummy.pkl",
            feature_columns=["f1"],
            target_column="y",
        )
        # 1 log with high latency
        PredictionLog.objects.create(
            trained_model=model,
            input_data={"f1": 1.0},
            prediction_result="0",
            response_time_ms=800.0
        )

        client = Client()
        response = client.get(f'/monitor/{model.model_name}/')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['latency_alert'])
        self.assertFalse(response.context['drift_alert'])
        self.assertEqual(response.context['drift_report']['status'], 'INSUFFICIENT_DATA')
