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
            
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return render(request, 'automl/build.html', {'error': f'Error reading CSV: {str(e)}'})
            
        df = df.dropna()
        if df.empty:
            return render(request, 'automl/build.html', {'error': 'Dataset is empty after dropping missing values'})
            
        target_col = df.columns[-1]
        feature_cols = list(df.columns[:-1])
        
        # ensure target is discrete string for classification
        df[target_col] = df[target_col].astype(str)
        
        X = df[feature_cols]
        # Ignore CustomerID if present, it's just an index usually
        X = X.drop(columns=['CustomerID'], errors='ignore')
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
    return render(request, 'automl/dashboard.html', {'models': models})

def predict(request, model_name):
    model_obj = get_object_or_404(TrainedModel, model_name=model_name)
    features = model_obj.feature_columns
    
    if request.method == 'POST':
        start_time = time.time()
        input_data = {}
        for f in features:
            val = request.POST.get(f)
            try:
                input_data[f] = float(val) if val else 0.0
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
            
            PredictionLog.objects.create(
                trained_model=model_obj,
                input_data=input_data,
                prediction_result=str(prediction),
                response_time_ms=response_time_ms
            )
            
            return render(request, 'automl/predict.html', {
                'model': model_obj, 
                'features': features, 
                'prediction': prediction,
                'response_time_ms': round(response_time_ms, 2)
            })
        except Exception as e:
            return render(request, 'automl/predict.html', {
                'model': model_obj, 
                'features': features, 
                'error': f'Prediction failed: {str(e)}'
            })
            
    return render(request, 'automl/predict.html', {'model': model_obj, 'features': features})

def monitor(request, model_name):
    model_obj = get_object_or_404(TrainedModel, model_name=model_name)
    logs = model_obj.logs.all().order_by('-timestamp')
    
    total_predictions = logs.count()
    avg_response_time = logs.aggregate(Avg('response_time_ms'))['response_time_ms__avg'] or 0
    recent_logs = logs[:10]
    
    # 1. Latency SLA Alert (System / DevOps performance)
    latency_alert = bool(total_predictions > 0 and avg_response_time > 500)
    
    # 2. Statistical Data Drift Detection (MLOps / Statistical feature distribution shift)
    drift_report = detect_data_drift(model_obj, sample_window=100, alpha=0.05, min_samples=25)
    drift_alert = bool(drift_report.get('drift_detected', False))
        
    logs_for_chart = list(recent_logs[:20])
    chart_labels = [log.timestamp.strftime("%H:%M:%S") for log in reversed(logs_for_chart)]
    chart_data = [log.response_time_ms for log in reversed(logs_for_chart)]

    # Fetch dataset statistics
    dataset_stats = model_obj.dataset_stats
    dataset_path = model_obj.get_absolute_dataset_path()
    if not dataset_stats and dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            dataset_stats = compute_dataset_stats(df, target_col=model_obj.target_column)
            model_obj.dataset_stats = dataset_stats
            model_obj.save(update_fields=['dataset_stats'])
        except Exception:
            dataset_stats = None

    context = {
        'model': model_obj,
        'total_predictions': total_predictions,
        'avg_response_time': round(avg_response_time, 2),
        'recent_logs': recent_logs,
        'latency_alert': latency_alert,
        'drift_alert': drift_alert,
        'drift_report': drift_report,
        'chart_labels': chart_labels,
        'chart_data': chart_data,
        'dataset_stats': dataset_stats,
    }
    return render(request, 'automl/monitor.html', context)

