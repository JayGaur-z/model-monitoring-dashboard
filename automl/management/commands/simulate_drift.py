import os
import random
import time
import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from automl.models import TrainedModel, PredictionLog
from automl.drift import detect_data_drift

class Command(BaseCommand):
    help = "Simulates normal vs deliberately shifted inference stream to demonstrate dynamic data drift detection."

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            '--model-name',
            dest='model',
            type=str,
            default='Loan_Approval_Model',
            help='Name of the trained model to simulate for (default: Loan_Approval_Model)'
        )
        parser.add_argument(
            '--mode',
            type=str,
            choices=['normal', 'drifted'],
            default='normal',
            help="Simulation mode: 'normal' draws from baseline distribution; 'drifted' injects deliberate feature shift."
        )
        parser.add_argument(
            '--count',
            type=int,
            default=50,
            help='Number of inference records to simulate (default: 50, minimum: 25)'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing prediction logs for this model before running simulation'
        )

    def handle(self, *args, **options):
        model_name = options['model']
        mode = options['mode']
        count = max(options['count'], 25)
        clear_existing = options['clear_existing']

        try:
            model_obj = TrainedModel.objects.get(model_name=model_name)
        except TrainedModel.DoesNotExist:
            raise CommandError(f"Model '{model_name}' not found in database. Train it first.")

        dataset_path = model_obj.get_absolute_dataset_path()
        if not dataset_path or not os.path.exists(dataset_path):
            raise CommandError(f"Reference dataset not found for model '{model_name}'.")

        pkl_path = model_obj.get_absolute_pkl_path()
        if not pkl_path or not os.path.exists(pkl_path):
            raise CommandError(f"Model artifact (.pkl) not found at '{pkl_path}'.")

        clf = joblib.load(pkl_path)
        feature_cols = model_obj.feature_columns

        if clear_existing:
            deleted_count, _ = model_obj.logs.all().delete()
            self.stdout.write(self.style.WARNING(f"Cleared {deleted_count} existing logs for '{model_name}'."))

        df_ref = pd.read_csv(dataset_path)
        df_ref.columns = df_ref.columns.str.strip()
        df_ref = df_ref.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

        # Drop non-feature columns
        target_col = model_obj.target_column
        cols_to_drop = [c for c in [target_col, 'loan_id', 'CustomerID', 'id'] if c in df_ref.columns]
        df_features = df_ref.drop(columns=cols_to_drop, errors='ignore')

        if any(not pd.api.types.is_numeric_dtype(df_features[c]) for c in df_features.columns):
            df_encoded = pd.get_dummies(df_features, drop_first=True)
        else:
            df_encoded = df_features

        # Ensure all model features exist in encoded reference
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0

        df_encoded = df_encoded[feature_cols]

        self.stdout.write(self.style.NOTICE(
            f"\nGenerating {count} simulated inference records in '{mode.upper()}' mode for '{model_name}'..."
        ))

        # Sample base rows
        sample_df = df_encoded.sample(n=count, replace=True, random_state=42).copy()

        if mode == 'drifted':
            self.stdout.write(self.style.WARNING(
                "Injecting Deliberate Covariate Shift (Data Drift):\n"
                "  1. cibil_score: Shifted from normal baseline (mean ~600) to subprime distribution (mean ~380).\n"
                "  2. income_annum: Reduced by 50% to simulate macroeconomic downturn.\n"
                "  3. loan_amount: Increased by 40% (surge in debt demand).\n"
                "  4. self_employed_ Yes: Increased to 85%."
            ))
            if 'cibil_score' in sample_df.columns:
                # Subprime shift: mean 380, std 40, clipped to [300, 550]
                sample_df['cibil_score'] = np.clip(np.random.normal(380, 40, size=count), 300, 550).round()
            if 'income_annum' in sample_df.columns:
                # 50% income drop
                sample_df['income_annum'] = (sample_df['income_annum'] * 0.50).round()
            if 'loan_amount' in sample_df.columns:
                # 40% loan surge
                sample_df['loan_amount'] = (sample_df['loan_amount'] * 1.40).round()
            if 'self_employed_ Yes' in sample_df.columns:
                # Shift self_employed proportion to 85%
                sample_df['self_employed_ Yes'] = np.random.choice([0, 1], size=count, p=[0.15, 0.85])

        # Run inference and record logs
        new_logs = []
        for _, row in sample_df.iterrows():
            row_dict = row.to_dict()
            row_df = pd.DataFrame([row_dict])[feature_cols]

            start_t = time.time()
            pred = clf.predict(row_df)[0]
            latency = (time.time() - start_t) * 1000 + random.uniform(8, 25)

            new_logs.append(PredictionLog(
                trained_model=model_obj,
                input_data=row_dict,
                prediction_result=str(pred),
                response_time_ms=latency
            ))

        PredictionLog.objects.bulk_create(new_logs)
        self.stdout.write(self.style.SUCCESS(f"Successfully recorded {len(new_logs)} prediction logs."))

        # Run dynamic drift evaluation
        report = detect_data_drift(model_obj, sample_window=count, alpha=0.05, min_samples=25)

        self.stdout.write("=" * 70)
        self.stdout.write(self.style.SUCCESS(f"DRIFT DETECTION AUDIT REPORT ({mode.upper()} MODE)"))
        self.stdout.write("=" * 70)
        self.stdout.write(f"Sample Count Evaluated: {report['sample_count']}")
        self.stdout.write(f"Significance Level (alpha): {report['alpha']}")
        self.stdout.write(f"Overall Drift Detected: {report['drift_detected']}")
        self.stdout.write(f"Summary Message: {report['message']}")
        self.stdout.write("-" * 70)
        self.stdout.write(f"{'Feature':<28} {'Test Type':<18} {'Statistic':<10} {'p-value':<10} {'Status':<12}")
        self.stdout.write("-" * 70)

        for f in report.get('feature_results', []):
            status_str = "DRIFT!" if f['drift_detected'] else "STABLE"
            self.stdout.write(
                f"{f['feature']:<28} {f['test']:<18} {str(f['statistic']):<10} {str(f['p_value']):<10} {status_str:<12}"
            )
        self.stdout.write("=" * 70)
