import os
from django.conf import settings
from django.db import models

class TrainedModel(models.Model):
    model_name = models.CharField(max_length=255, unique=True)
    pkl_file_path = models.CharField(max_length=500)
    feature_columns = models.JSONField()
    target_column = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    dataset_file_path = models.CharField(max_length=500, blank=True, null=True)
    dataset_stats = models.JSONField(blank=True, null=True)

    def __str__(self):
        return self.model_name

    def get_absolute_pkl_path(self):
        """Resolves the .pkl model file path portably across environments."""
        if self.pkl_file_path and os.path.exists(self.pkl_file_path):
            return self.pkl_file_path
        if self.pkl_file_path:
            filename = os.path.basename(self.pkl_file_path)
            local_path = os.path.join(settings.BASE_DIR, 'saved_models', filename)
            if os.path.exists(local_path):
                return local_path
        return self.pkl_file_path

    def get_absolute_dataset_path(self):
        """Resolves the dataset CSV file path portably across environments."""
        if self.dataset_file_path and os.path.exists(self.dataset_file_path):
            return self.dataset_file_path
        if self.dataset_file_path:
            filename = os.path.basename(self.dataset_file_path)
            local_path = os.path.join(settings.BASE_DIR, 'saved_datasets', filename)
            if os.path.exists(local_path):
                return local_path
        # Smart fallback: inspect saved_datasets for matching dataset CSV
        datasets_dir = os.path.join(settings.BASE_DIR, 'saved_datasets')
        if os.path.exists(datasets_dir):
            clean_name = self.model_name.lower().replace('_', ' ').replace('-', ' ')
            tokens = [t for t in clean_name.split() if t not in ['model', 'prediction', 'predictor']]
            if tokens:
                for fname in os.listdir(datasets_dir):
                    if fname.endswith('.csv'):
                        fname_clean = fname.lower()
                        if all(t in fname_clean for t in tokens):
                            return os.path.join(datasets_dir, fname)
        return None

class PredictionLog(models.Model):
    trained_model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='logs')
    input_data = models.JSONField()
    prediction_result = models.CharField(max_length=255)
    response_time_ms = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.trained_model.model_name} Prediction at {self.timestamp}"
