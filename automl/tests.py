import os
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

    def _get_or_create_loan_model(self):
        model = TrainedModel.objects.filter(model_name="Loan_Approval_Model").first()
        if not model:
            feature_cols = [
                'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                'luxury_assets_value', 'bank_asset_value', 'education_ Not Graduate', 'self_employed_ Yes'
            ]
            model = TrainedModel.objects.create(
                model_name="Loan_Approval_Model",
                pkl_file_path=os.path.join('saved_models', 'Loan_Approval_Model.pkl'),
                dataset_file_path=os.path.join('saved_datasets', 'loan_approval_dataset.csv'),
                feature_columns=feature_cols,
                target_column='loan_status'
            )
        return model

    def test_loan_approval_model_pipeline_and_prediction(self):
        """Verifies Loan_Approval_Model loads cleanly from disk and executes predictions without errors."""
        model = self._get_or_create_loan_model()
        pkl_path = model.get_absolute_pkl_path()
        self.assertTrue(os.path.exists(pkl_path), f"Model file must exist at {pkl_path}")

        import joblib
        clf = joblib.load(pkl_path)
        sample_input = {f: 1.0 for f in model.feature_columns}
        input_df = pd.DataFrame([sample_input])[model.feature_columns]
        pred = clf.predict(input_df)[0]
        self.assertIn(str(pred), ['0', '1', '0.0', '1.0'])

    def test_simulate_drift_command_detects_deliberate_shift(self):
        """Verifies simulate_drift command generates stable outputs in normal mode and detects drift in drifted mode."""
        from django.core.management import call_command
        from io import StringIO

        model = self._get_or_create_loan_model()

        out = StringIO()
        # Test Normal Mode
        call_command('simulate_drift', model='Loan_Approval_Model', mode='normal', count=40, clear_existing=True, stdout=out)
        report_normal = detect_data_drift(model, sample_window=40)
        self.assertEqual(report_normal['status'], 'SUCCESS')
        self.assertFalse(report_normal['drift_detected'], "Normal mode should not trigger overall drift alert")

        # Test Drifted Mode
        call_command('simulate_drift', model='Loan_Approval_Model', mode='drifted', count=40, clear_existing=True, stdout=out)
        report_drifted = detect_data_drift(model, sample_window=40)
        self.assertEqual(report_drifted['status'], 'SUCCESS')
        self.assertTrue(report_drifted['drift_detected'], "Drifted mode must detect simulated feature drift")
        
        # Verify specific shifted features were flagged
        drifted_feature_names = [f['feature'] for f in report_drifted['feature_results'] if f['drift_detected']]
        self.assertIn('cibil_score', drifted_feature_names)
        self.assertIn('income_annum', drifted_feature_names)

    def test_simulate_drift_isolation_preserves_other_models(self):
        """Verifies simulate_drift --clear-existing only clears the target model logs without touching other models."""
        from django.core.management import call_command
        from io import StringIO

        other_model = TrainedModel.objects.create(
            model_name="Untouched_Peer_Model",
            pkl_file_path="dummy.pkl",
            feature_columns=["f1"],
            target_column="y"
        )
        # Create 5 logs for other model
        for i in range(5):
            PredictionLog.objects.create(
                trained_model=other_model,
                input_data={"f1": float(i)},
                prediction_result="0",
                response_time_ms=15.0
            )

        loan_model = self._get_or_create_loan_model()
        out = StringIO()
        # Run simulate_drift with --clear-existing for loan_model
        call_command('simulate_drift', model='Loan_Approval_Model', mode='normal', count=30, clear_existing=True, stdout=out)

        # Confirm peer model logs were NOT touched
        self.assertEqual(PredictionLog.objects.filter(trained_model=other_model).count(), 5)
        self.assertEqual(PredictionLog.objects.filter(trained_model=loan_model).count(), 30)

    def test_settings_and_production_environment_configuration(self):
        """Verifies Django settings support configurable environments (DB path, static root, WSGI)."""
        from django.conf import settings
        self.assertTrue(hasattr(settings, 'STATIC_ROOT'))
        self.assertTrue(hasattr(settings, 'WSGI_APPLICATION'))
        self.assertEqual(settings.WSGI_APPLICATION, 'core_project.wsgi.application')
        self.assertEqual(settings.DATABASES['default']['ENGINE'], 'django.db.backends.sqlite3')

    def test_drift_detection_includes_and_evaluates_categorical_features(self):
        """Verifies detect_data_drift includes and evaluates categorical features using Chi-Square."""
        model = self._get_or_create_loan_model()
        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('simulate_drift', model='Loan_Approval_Model', mode='drifted', count=30, clear_existing=True, stdout=out)
        report = detect_data_drift(model, sample_window=30)
        self.assertEqual(report['status'], 'SUCCESS')
        tested_features = {f['feature']: f['test'] for f in report['feature_results']}
        # Ensure categorical features are not skipped
        self.assertIn('education_ Not Graduate', tested_features)
        self.assertIn('self_employed_ Yes', tested_features)
        self.assertEqual(tested_features['education_ Not Graduate'], 'Chi-Square')
        self.assertEqual(tested_features['self_employed_ Yes'], 'Chi-Square')

    def test_build_model_rejects_duplicate_model_name(self):
        """Verifies build_model rejects duplicate model names with 200 and error message without crashing with IntegrityError."""
        model = self._get_or_create_loan_model()
        client = Client()
        from django.core.files.uploadedfile import SimpleUploadedFile
        csv_content = b"feature1,feature2,target\n1,2,0\n3,4,1\n"
        csv_file = SimpleUploadedFile("test.csv", csv_content, content_type="text/csv")
        response = client.post(reverse('build_model'), {
            'model_name': model.model_name,
            'csv_file': csv_file
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('already exists', response.context['error'])
        self.assertEqual(response.context['model_name'], model.model_name)

    def test_predict_handles_empty_feature_model_gracefully(self):
        """Verifies predict view returns a clean error when a model has no feature columns."""
        model = TrainedModel.objects.create(
            model_name="Empty_Feature_Model",
            pkl_file_path="dummy.pkl",
            feature_columns=[],
            target_column="target"
        )
        client = Client()
        response = client.get(reverse('predict', kwargs={'model_name': model.model_name}))
        self.assertEqual(response.status_code, 200)
        self.assertIn('error', response.context)
        self.assertIn('no registered feature columns', response.context['error'])

    def test_portable_path_resolver_handles_settings_base_dir(self):
        """Verifies model path resolver methods anchor relative paths to settings.BASE_DIR."""
        from django.conf import settings
        model = TrainedModel(
            model_name="Path_Test_Model",
            pkl_file_path="saved_models/test.pkl",
            dataset_file_path="saved_datasets/test.csv"
        )
        expected_pkl = os.path.normpath(os.path.join(settings.BASE_DIR, 'saved_models', 'test.pkl'))
        expected_csv = os.path.normpath(os.path.join(settings.BASE_DIR, 'saved_datasets', 'test.csv'))
        self.assertEqual(os.path.normpath(model.get_absolute_pkl_path()), expected_pkl)
        self.assertEqual(os.path.normpath(model.get_absolute_dataset_path()), expected_csv)


