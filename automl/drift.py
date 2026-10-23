import os
import numpy as np
import pandas as pd
from scipy import stats
from django.conf import settings

def compute_numerical_drift(reference_series, production_series, alpha=0.05):
    """
    Computes Two-Sample Kolmogorov-Smirnov (KS) test for a numerical feature.
    
    Hypotheses:
      - Null Hypothesis (H0): Reference and production samples come from the same continuous distribution.
      - Alternative Hypothesis (H1): Production sample distribution differs significantly from reference.
      
    Decision Rule:
      - If p-value < alpha (default 0.05): Reject H0 -> Drift detected.
      - If p-value >= alpha: Fail to reject H0 -> Stable distribution.
    """
    ref_clean = pd.to_numeric(reference_series, errors='coerce').dropna()
    prod_clean = pd.to_numeric(production_series, errors='coerce').dropna()

    if len(ref_clean) < 5 or len(prod_clean) < 5:
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': None,
            'p_value': None,
            'drift_detected': False,
            'status': 'SKIPPED_INSUFFICIENT_DATA',
            'message': f'Fewer than 5 non-null values (ref: {len(ref_clean)}, prod: {len(prod_clean)})'
        }

    # Constant values check (zero variance)
    if ref_clean.nunique() <= 1 and prod_clean.nunique() <= 1:
        is_same = bool(ref_clean.iloc[0] == prod_clean.iloc[0])
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': 0.0 if is_same else 1.0,
            'p_value': 1.0 if is_same else 0.0,
            'drift_detected': not is_same,
            'status': 'CONSTANT_VALUES',
            'message': 'Constant values in both reference and production'
        }

    res = stats.ks_2samp(ref_clean, prod_clean)
    statistic = float(res.statistic)
    p_value = float(res.pvalue)
    drift_detected = bool(p_value < alpha)

    return {
        'test': 'Kolmogorov-Smirnov',
        'statistic': round(statistic, 4),
        'p_value': round(p_value, 4),
        'drift_detected': drift_detected,
        'status': 'SUCCESS',
        'message': f'Drift detected (p={p_value:.4f} < {alpha})' if drift_detected else f'Stable (p={p_value:.4f} >= {alpha})'
    }

def compute_categorical_drift(reference_series, production_series, alpha=0.05):
    """
    Computes Chi-Square Test of Independence on a 2 x K contingency table.
    
    Hypotheses:
      - Null Hypothesis (H0): Category distribution in production matches the reference baseline.
      - Alternative Hypothesis (H1): Category distribution in production differs significantly.
      
    Decision Rule:
      - If p-value < alpha (default 0.05): Reject H0 -> Drift detected.
      - If p-value >= alpha: Fail to reject H0 -> Stable distribution.
    """
    ref_clean = reference_series.dropna().astype(str)
    prod_clean = production_series.dropna().astype(str)

    if len(ref_clean) < 5 or len(prod_clean) < 5:
        return {
            'test': 'Chi-Square',
            'statistic': None,
            'p_value': None,
            'drift_detected': False,
            'status': 'SKIPPED_INSUFFICIENT_DATA',
            'message': f'Fewer than 5 non-null values (ref: {len(ref_clean)}, prod: {len(prod_clean)})'
        }

    all_categories = sorted(list(set(ref_clean.unique()).union(set(prod_clean.unique()))))
    if len(all_categories) < 2:
        return {
            'test': 'Chi-Square',
            'statistic': 0.0,
            'p_value': 1.0,
            'drift_detected': False,
            'status': 'SINGLE_CATEGORY',
            'message': 'Only one distinct category observed'
        }

    ref_counts = ref_clean.value_counts()
    prod_counts = prod_clean.value_counts()

    # 2 x K Contingency Table
    row_ref = [ref_counts.get(cat, 0) for cat in all_categories]
    row_prod = [prod_counts.get(cat, 0) for cat in all_categories]
    table = [row_ref, row_prod]

    try:
        chi2, p_val, dof, _ = stats.chi2_contingency(table)
        drift_detected = bool(p_val < alpha)
        return {
            'test': 'Chi-Square',
            'statistic': round(float(chi2), 4),
            'p_value': round(float(p_val), 4),
            'drift_detected': drift_detected,
            'status': 'SUCCESS',
            'message': f'Drift detected (p={p_val:.4f} < {alpha})' if drift_detected else f'Stable (p={p_val:.4f} >= {alpha})'
        }
    except Exception as e:
        return {
            'test': 'Chi-Square',
            'statistic': None,
            'p_value': None,
            'drift_detected': False,
            'status': 'TEST_ERROR',
            'message': f'Calculation error: {str(e)}'
        }

def detect_data_drift(model_obj, sample_window=100, alpha=0.05, min_samples=25):
    """
    Detects data drift by comparing recent prediction input logs against the baseline reference dataset.
    
    Guards & Assumptions:
      1. Minimum Sample Size: Requires at least `min_samples` logs in the evaluation window.
      2. Valid Reference Dataset: Requires a readable reference CSV.
      3. Feature Alignment: Evaluates all model feature columns against baseline distributions.
      4. Ground-Truth Awareness: Notes that input data drift (covariate shift) indicates distribution change
         in P(X), but ground-truth labels (Y) are needed to prove prediction performance or accuracy degradation.
    """
    disclaimer = (
        "Data Drift (Covariate Shift) measures statistical changes in input feature distributions P(X). "
        "It serves as a leading indicator of risk, but does not prove accuracy degradation without ground-truth labels (Y)."
    )

    logs = model_obj.logs.all().order_by('-timestamp')[:sample_window]
    total_logs_count = logs.count()

    if total_logs_count < min_samples:
        return {
            'status': 'INSUFFICIENT_DATA',
            'drift_detected': False,
            'sample_count': total_logs_count,
            'min_required': min_samples,
            'sample_window': sample_window,
            'alpha': alpha,
            'message': f"Insufficient inference data: collected {total_logs_count} samples (minimum {min_samples} required for statistical validity).",
            'feature_results': [],
            'drifted_features_count': 0,
            'total_tested_features': 0,
            'performance_disclaimer': disclaimer,
        }

    dataset_path = model_obj.get_absolute_dataset_path()
    if not dataset_path or not os.path.exists(dataset_path):
        return {
            'status': 'MISSING_REFERENCE_DATA',
            'drift_detected': False,
            'sample_count': total_logs_count,
            'min_required': min_samples,
            'sample_window': sample_window,
            'alpha': alpha,
            'message': "Reference baseline dataset file could not be found on disk.",
            'feature_results': [],
            'drifted_features_count': 0,
            'total_tested_features': 0,
            'performance_disclaimer': disclaimer,
        }

    try:
        df_ref = pd.read_csv(dataset_path)
    except Exception as e:
        return {
            'status': 'ERROR_READING_DATASET',
            'drift_detected': False,
            'sample_count': total_logs_count,
            'min_required': min_samples,
            'sample_window': sample_window,
            'alpha': alpha,
            'message': f"Failed to load reference dataset: {str(e)}",
            'feature_results': [],
            'drifted_features_count': 0,
            'total_tested_features': 0,
            'performance_disclaimer': disclaimer,
        }

    # Extract production data from logs
    prod_records = [log.input_data for log in logs if isinstance(log.input_data, dict)]
    if len(prod_records) < min_samples:
        return {
            'status': 'INSUFFICIENT_DATA',
            'drift_detected': False,
            'sample_count': len(prod_records),
            'min_required': min_samples,
            'sample_window': sample_window,
            'alpha': alpha,
            'message': f"Valid production logs ({len(prod_records)}) below threshold ({min_samples}).",
            'feature_results': [],
            'drifted_features_count': 0,
            'total_tested_features': 0,
            'performance_disclaimer': disclaimer,
        }

    df_prod = pd.DataFrame(prod_records)

    # Clean reference features: drop target and CustomerID if present
    target_col = model_obj.target_column
    cols_to_drop = [c for c in [target_col, 'CustomerID', 'customerID'] if c in df_ref.columns]
    df_ref_features = df_ref.drop(columns=cols_to_drop, errors='ignore')

    # If reference data has raw categorical columns but model uses one-hot dummy columns,
    # expand reference features to match
    if any(df_ref_features[c].dtype == 'object' for c in df_ref_features.columns):
        df_ref_encoded = pd.get_dummies(df_ref_features, drop_first=True)
    else:
        df_ref_encoded = df_ref_features

    feature_results = []
    features_to_test = model_obj.feature_columns if model_obj.feature_columns else list(df_prod.columns)

    for feature in features_to_test:
        if feature not in df_prod.columns:
            continue

        prod_series = df_prod[feature]

        # Find matching series in reference
        if feature in df_ref_encoded.columns:
            ref_series = df_ref_encoded[feature]
        elif feature in df_ref_features.columns:
            ref_series = df_ref_features[feature]
        else:
            continue

        # Decide whether feature is continuous numerical or categorical
        # If numeric dtype with >= 10 unique values -> KS test
        # If categorical or numeric with < 10 unique values (e.g. binary flags, discrete counts) -> Chi-Square
        is_num = pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(prod_series)
        unique_count = max(ref_series.nunique(), prod_series.nunique())

        if is_num and unique_count >= 10:
            result = compute_numerical_drift(ref_series, prod_series, alpha=alpha)
            test_type = 'Kolmogorov-Smirnov (Numerical)'
        else:
            result = compute_categorical_drift(ref_series, prod_series, alpha=alpha)
            test_type = 'Chi-Square (Categorical)'

        result['feature'] = feature
        result['test_type'] = test_type
        result['ref_count'] = int(ref_series.dropna().count())
        result['prod_count'] = int(prod_series.dropna().count())
        feature_results.append(result)

    drifted_count = sum(1 for f in feature_results if f.get('drift_detected', False))
    overall_drift = drifted_count > 0

    return {
        'status': 'SUCCESS',
        'drift_detected': overall_drift,
        'sample_count': len(df_prod),
        'min_required': min_samples,
        'sample_window': sample_window,
        'alpha': alpha,
        'message': f"{drifted_count} out of {len(feature_results)} feature(s) show statistical drift." if overall_drift else "All tested features are statistically stable within baseline distribution.",
        'feature_results': feature_results,
        'drifted_features_count': drifted_count,
        'total_tested_features': len(feature_results),
        'performance_disclaimer': disclaimer,
    }
