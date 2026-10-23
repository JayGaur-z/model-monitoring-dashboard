import os
import time
import pandas as pd
import joblib
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import TrainedModel, PredictionLog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from django.db.models import Avg
from .stats import compute_dataset_stats
from .drift import detect_data_drift

def build_model(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        csv_file = request.FILES.get('csv_file')
        
        if not model_name or not csv_file:
            return render(request, 'automl/build.html', {'error': 'Please provide model name and CSV file'})

        model_name = model_name.strip()
        if TrainedModel.objects.filter(model_name=model_name).exists():
            return render(request, 'automl/build.html', {
                'error': f'A model with the identifier "{model_name}" already exists. Please choose a unique name.',
                'model_name': model_name
            })
            
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return render(request, 'automl/build.html', {
                'error': f'Error reading CSV: {str(e)}',
                'model_name': model_name
            })
            
        df = df.dropna()
        if df.empty:
            return render(request, 'automl/build.html', {
                'error': 'Dataset is empty after dropping missing values',
                'model_name': model_name
            })
            
        target_col = df.columns[-1]
        feature_cols = list(df.columns[:-1])
        
        # ensure target is discrete string for classification
        df[target_col] = df[target_col].astype(str)
        
        X = df[feature_cols]
        # Ignore ID columns if present (e.g. CustomerID, loan_id)
        X = X.drop(columns=['CustomerID', 'loan_id', 'Loan_ID', 'id', 'ID'], errors='ignore')
        y = df[target_col]
        
        # Basic encoding for categorical
        X = pd.get_dummies(X, drop_first=True)
        final_feature_cols = list(X.columns)
        
        le = LabelEncoder()
        y = le.fit_transform(y)
            
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        models_dir = os.path.join(settings.BASE_DIR, 'saved_models')
        os.makedirs(models_dir, exist_ok=True)
        
        pkl_path = os.path.join(models_dir, f"{model_name}.pkl")
        joblib.dump(clf, pkl_path)
        
        # Save original dataset
        datasets_dir = os.path.join(settings.BASE_DIR, 'saved_datasets')
        os.makedirs(datasets_dir, exist_ok=True)
        dataset_path = os.path.join(datasets_dir, f"{model_name}.csv")
        df.to_csv(dataset_path, index=False)

        # Compute Pandas statistical summary
        stats = compute_dataset_stats(df, target_col=target_col)

        TrainedModel.objects.create(
            model_name=model_name,
            pkl_file_path=pkl_path,
            feature_columns=final_feature_cols,
            target_column=target_col,
            dataset_file_path=dataset_path,
            dataset_stats=stats,
        )
        return redirect('dashboard')
        
    return render(request, 'automl/build.html')

def dashboard(request):
    models = TrainedModel.objects.all().order_by('-created_at')
    total_models_count = models.count()
    total_predictions_count = PredictionLog.objects.count()
    system_avg_latency = PredictionLog.objects.aggregate(Avg('response_time_ms'))['response_time_ms__avg'] or 0

    model_cards = []
    active_drift_alerts_count = 0
    for m in models:
        pred_count = m.logs.count()
        m_avg_latency = m.logs.aggregate(Avg('response_time_ms'))['response_time_ms__avg'] or 0
        drift_info = detect_data_drift(m, sample_window=50, min_samples=25)
        if drift_info.get('drift_detected', False):
            active_drift_alerts_count += 1
        model_cards.append({
            'obj': m,
            'prediction_count': pred_count,
            'avg_latency': round(m_avg_latency, 2),
            'drift_status': drift_info.get('status', 'NO_DATA'),
            'drift_detected': drift_info.get('drift_detected', False),
            'drift_message': drift_info.get('message', ''),
            'drifted_count': drift_info.get('drifted_features_count', 0),
            'total_features': len(m.feature_columns) if m.feature_columns else 0
        })

    recent_global_logs = PredictionLog.objects.select_related('trained_model').order_by('-timestamp')[:8]

    context = {
        'models': models,
        'model_cards': model_cards,
        'total_models_count': total_models_count,
        'total_predictions_count': total_predictions_count,
        'system_avg_latency': round(system_avg_latency, 2),
        'active_drift_alerts_count': active_drift_alerts_count,
        'recent_global_logs': recent_global_logs,
    }
    return render(request, 'automl/dashboard.html', context)

def predict(request, model_name):
    model_obj = get_object_or_404(TrainedModel, model_name=model_name)
    features = model_obj.feature_columns
    is_loan_model = (model_name == 'Loan_Approval_Model')
    
    if not features:
        return render(request, 'automl/predict.html', {
            'model': model_obj,
            'features': [],
            'is_loan_model': is_loan_model,
            'error': 'This model has no registered feature columns and cannot execute inference.'
        })
        
    if request.method == 'POST':
        start_time = time.time()
        input_data = {}

        # Handle friendly categorical dropdown values
        friendly_education = request.POST.get('friendly_education')
        friendly_self_employed = request.POST.get('friendly_self_employed')

        for f in features:
            if f == 'education_ Not Graduate' and friendly_education:
                input_data[f] = 1.0 if friendly_education == 'Not Graduate' else 0.0
            elif f == 'self_employed_ Yes' and friendly_self_employed:
                input_data[f] = 1.0 if friendly_self_employed == 'Yes' else 0.0
            else:
                val = request.POST.get(f)
                try:
                    input_data[f] = float(val) if val is not None and val != '' else 0.0
                except ValueError:
                    input_data[f] = 0.0
                
        try:
            clf = joblib.load(model_obj.get_absolute_pkl_path())
            input_df = pd.DataFrame([input_data])
            # Reorder columns to match training exactly
            input_df = input_df[features]
            prediction = clf.predict(input_df)[0]
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            prediction_str = str(prediction)
            friendly_label = prediction_str
            is_positive_outcome = True

            if is_loan_model:
                if str(prediction) in ['0', '0.0', 'Approved', ' Approved']:
                    friendly_label = 'Approved'
                    is_positive_outcome = True
                else:
                    friendly_label = 'Rejected'
                    is_positive_outcome = False
            elif 'churn' in model_name.lower():
                if str(prediction) in ['1', '1.0', 'Churn', '1']:
                    friendly_label = 'High Churn Risk'
                    is_positive_outcome = False
                else:
                    friendly_label = 'Retained (Low Risk)'
                    is_positive_outcome = True
            
            PredictionLog.objects.create(
                trained_model=model_obj,
                input_data=input_data,
                prediction_result=friendly_label,
                response_time_ms=response_time_ms
            )

            submitted_dict = dict(input_data)
            if is_loan_model:
                submitted_dict['friendly_education'] = friendly_education or ('Not Graduate' if input_data.get('education_ Not Graduate') == 1.0 else 'Graduate')
                submitted_dict['friendly_self_employed'] = friendly_self_employed or ('Yes' if input_data.get('self_employed_ Yes') == 1.0 else 'No')
            
            return render(request, 'automl/predict.html', {
                'model': model_obj, 
                'features': features,
                'is_loan_model': is_loan_model,
                'prediction': friendly_label,
                'raw_prediction': prediction_str,
                'is_positive_outcome': is_positive_outcome,
                'response_time_ms': round(response_time_ms, 2),
                'submitted_inputs': submitted_dict,
            })
        except Exception as e:
            return render(request, 'automl/predict.html', {
                'model': model_obj, 
                'features': features,
                'is_loan_model': is_loan_model,
                'error': f'Prediction failed: {str(e)}'
            })
            
    return render(request, 'automl/predict.html', {
        'model': model_obj,
        'features': features,
        'is_loan_model': is_loan_model
    })

def monitor(request, model_name):
    model_obj = get_object_or_404(TrainedModel, model_name=model_name)
    logs = model_obj.logs.all().order_by('-timestamp')
    
    total_predictions = logs.count()
    avg_response_time = logs.aggregate(Avg('response_time_ms'))['response_time_ms__avg'] or 0
    recent_logs = logs[:10]

    # Compute P95 latency (safely handling None/missing values)
    all_latencies = sorted([float(log.response_time_ms) for log in logs[:100] if log.response_time_ms is not None])
    p95_latency = all_latencies[int(len(all_latencies) * 0.95)] if all_latencies else avg_response_time
    
    # 1. Latency SLA Alert (System / DevOps performance)
    latency_alert = bool(total_predictions > 0 and avg_response_time > 500)
    
    # 2. Statistical Data Drift Detection (MLOps / Statistical feature distribution shift)
    drift_report = detect_data_drift(model_obj, sample_window=100, alpha=0.05, min_samples=25)
    drift_alert = bool(drift_report.get('drift_detected', False))
        
    logs_for_chart = list(recent_logs[:20])
    chart_labels = [log.timestamp.strftime("%H:%M:%S") for log in reversed(logs_for_chart)]
    chart_data = [round(log.response_time_ms, 2) for log in reversed(logs_for_chart)]

    # Fetch dataset statistics
    dataset_stats = model_obj.dataset_stats
    dataset_path = model_obj.get_absolute_dataset_path()
    df_raw = None
    if dataset_path and os.path.exists(dataset_path):
        try:
            df_raw = pd.read_csv(dataset_path)
            df_raw.columns = df_raw.columns.str.strip()
            if not dataset_stats:
                dataset_stats = compute_dataset_stats(df_raw, target_col=model_obj.target_column)
                model_obj.dataset_stats = dataset_stats
                model_obj.save(update_fields=['dataset_stats'])
        except Exception:
            dataset_stats = None

    # Compute Distribution Comparison Data for a primary numerical feature
    dist_feature_name = None
    dist_bins = []
    dist_ref_pct = []
    dist_prod_pct = []

    if df_raw is not None and total_predictions >= 10:
        # Choose target feature for distribution comparison
        candidate_features = ['cibil_score', 'income_annum', 'loan_amount', 'Age', 'Total Spend', 'area']
        for cand in candidate_features:
            if cand in df_raw.columns and cand in model_obj.feature_columns:
                dist_feature_name = cand
                break
        if not dist_feature_name and model_obj.feature_columns:
            dist_feature_name = model_obj.feature_columns[0]

        if dist_feature_name and dist_feature_name in df_raw.columns:
            try:
                ref_s = pd.to_numeric(df_raw[dist_feature_name], errors='coerce').dropna()
                prod_vals = [log.input_data.get(dist_feature_name) for log in logs[:100] if log.input_data and dist_feature_name in log.input_data]
                prod_s = pd.to_numeric(pd.Series(prod_vals), errors='coerce').dropna()

                if len(ref_s) >= 10 and len(prod_s) >= 5 and ref_s.nunique() > 1:
                    import numpy as np
                    min_val = min(float(ref_s.min()), float(prod_s.min()))
                    max_val = max(float(ref_s.max()), float(prod_s.max()))
                    bins = np.linspace(min_val, max_val, 7)
                    ref_counts, _ = np.histogram(ref_s, bins=bins)
                    prod_counts, _ = np.histogram(prod_s, bins=bins)
                    dist_bins = [f"{int(bins[i]):,}-{int(bins[i+1]):,}" for i in range(len(bins)-1)]
                    dist_ref_pct = [round(float(c / len(ref_s) * 100), 1) for c in ref_counts]
                    dist_prod_pct = [round(float(c / len(prod_s) * 100), 1) for c in prod_counts]
            except Exception:
                dist_feature_name = None

    context = {
        'model': model_obj,
        'total_predictions': total_predictions,
        'avg_response_time': round(avg_response_time, 2),
        'p95_latency': round(p95_latency, 2),
        'recent_logs': recent_logs,
        'latency_alert': latency_alert,
        'drift_alert': drift_alert,
        'drift_report': drift_report,
        'chart_labels': chart_labels,
        'chart_data': chart_data,
        'dataset_stats': dataset_stats,
        'dist_feature_name': dist_feature_name,
        'dist_bins': dist_bins,
        'dist_ref_pct': dist_ref_pct,
        'dist_prod_pct': dist_prod_pct,
    }
    return render(request, 'automl/monitor.html', context)

