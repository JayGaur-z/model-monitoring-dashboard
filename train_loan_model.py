import os
import django
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

from django.conf import settings
from automl.models import TrainedModel
from automl.stats import compute_dataset_stats

def train_and_register_loan_model():
    print("=" * 60)
    print("Training Loan Approval / Credit Risk Prediction Model")
    print("=" * 60)

    dataset_path = os.path.join(settings.BASE_DIR, 'saved_datasets', 'loan_approval_dataset.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    target_col = 'loan_status'
    print(f"Dataset shape: {df.shape} (Rows: {len(df)}, Features: {df.shape[1] - 1})")
    print(f"Target distribution:\n{df[target_col].value_counts().to_dict()}")

    # 1. Feature Preprocessing (Drop ID column)
    X_raw = df.drop(columns=['loan_id', target_col], errors='ignore')
    y_raw = df[target_col].astype(str)

    # Convert categorical features using get_dummies
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_columns = list(X.columns)
    print(f"Encoded feature columns ({len(feature_columns)}): {feature_columns}")

    # Encode target: Approved=1, Rejected=0
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 2. Train/Test Split (Prevent Data Leakage: 80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Train Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate Performance on Holdout Test Set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nModel Performance on Test Set:")
    print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # 5. Save Serialized Model (.pkl)
    models_dir = os.path.join(settings.BASE_DIR, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    model_name = "Loan_Approval_Model"
    pkl_filename = f"{model_name}.pkl"
    full_pkl_path = os.path.join(models_dir, pkl_filename)
    joblib.dump(clf, full_pkl_path)
    print(f"Saved trained model artifact to: {full_pkl_path}")

    # 6. Compute Baseline Dataset Statistics
    stats = compute_dataset_stats(df, target_col=target_col)

    # 7. Register in Database (Portable relative paths)
    rel_pkl_path = os.path.join('saved_models', pkl_filename)
    rel_dataset_path = os.path.join('saved_datasets', 'loan_approval_dataset.csv')

    model_obj, created = TrainedModel.objects.update_or_create(
        model_name=model_name,
        defaults={
            'pkl_file_path': rel_pkl_path,
            'dataset_file_path': rel_dataset_path,
            'feature_columns': feature_columns,
            'target_column': target_col,
            'dataset_stats': stats,
        }
    )

    action = "Created new" if created else "Updated existing"
    print(f"{action} TrainedModel record in DB: id={model_obj.id}, name='{model_obj.model_name}'")
    print("=" * 60)
    return model_obj

if __name__ == "__main__":
    train_and_register_loan_model()
