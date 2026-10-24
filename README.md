# Model Monitoring Dashboard

A Django application for training and registering classification models from CSV datasets, running individual predictions, and monitoring prediction latency and input-data drift.

## Features

- Train a scikit-learn Random Forest classifier and register it in the model catalog.
- Submit feature values through the prediction playground.
- Record prediction inputs, results, and response times.
- View per-model prediction counts, mean and P95 latency, recent predictions, and drift diagnostics.
- Compare selected live numeric-feature distributions with the training dataset.

> **Report limitation:** The application has no built-in report download or export button. Use the report outline below to write a report, and capture screenshots or use your browser's **Print → Save as PDF** option to preserve dashboard evidence. The app reports monitoring metrics; it does not calculate model accuracy, precision, recall, or F1 score.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later recommended
- pip
- A CSV dataset for training

The project dependencies are listed in `requirements.txt`. A virtual environment named `venv` is included in this project folder.

## Run locally on Windows

Open PowerShell in the project folder (`Model Monitoring Dashboard - PS Project`) and run:

```powershell
.\venv\Scripts\Activate.ps1
python manage.py migrate
python manage.py runserver
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser. Stop the development server with **Ctrl+C**.

If PowerShell does not allow activating the environment, run its Python directly:

```powershell
.\venv\Scripts\python.exe manage.py migrate
.\venv\Scripts\python.exe manage.py runserver
```

To create a new environment instead of using the included one:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Train and monitor a model

1. Open the dashboard and choose **Deploy New Model**.
2. Enter a unique model identifier and upload a CSV file with a header row.
3. The app removes rows with missing values. It uses the **last CSV column as the target** and all earlier columns as input features.
4. The app removes common identifier columns (`id`, `ID`, `CustomerID`, `loan_id`, and `Loan_ID`) from the input features, one-hot encodes categorical features, then trains a 100-tree Random Forest classifier.
5. After training, select **Monitor** for monitoring information or **Playground** to submit predictions. Each successful prediction is added to the model's log.
6. Return to the dashboard to see the model catalog and platform-level summary.

Use a dataset with a meaningful discrete target column (for example, `approved` / `rejected`). Ensure the last column is the intended target. For reliable predictions, enter values that match the training feature schema and use representative, non-sensitive data.

## Monitoring metric notes

- **Prediction count and latency** come from logged prediction requests.
- **Mean latency** is the average logged response time in milliseconds.
- **P95 latency** is calculated from up to the latest 100 logged response times.
- A **latency alert** is shown when mean latency is over 500 ms.
- **Data drift** compares recent prediction inputs with the saved training dataset. The monitor uses statistical tests at alpha 0.05 and needs at least 25 recent samples to evaluate drift.
- The distribution comparison chart needs saved reference data and at least 10 predictions; it is shown only when a suitable numeric feature can be compared.
- With too few predictions, the dashboard may show that the drift evaluation has insufficient data. Run more predictions before capturing drift results.

These values describe monitoring behavior, not predictive model quality. Do not describe a stable drift result as proof that the model is accurate.

## Creating a project report

Use this outline to prepare a report in your preferred document editor. Replace bracketed text with your own project details and observed results.

### 1. Title and objective

**Title:** Model Monitoring Dashboard  
**Objective:** [Explain what the application is designed to do and why deployed models need monitoring.]

### 2. Problem statement

[Describe the problem of tracking deployed machine-learning models, prediction activity, response time, and changes in incoming data.]

### 3. Technologies used

- Python and Django
- pandas and NumPy
- scikit-learn (Random Forest classifier)
- Joblib (model serialization)
- SQLite (application database)
- HTML/CSS/JavaScript templates for the dashboard

### 4. System workflow

Describe the flow: upload a labeled CSV → train and register a model → submit prediction inputs → log the result and response time → review model and drift monitoring on the dashboard.

### 5. Dataset and training

Record:

- Dataset name/source: [ ]
- Number of rows and columns: [ ]
- Input features: [ ]
- Target column (must be the last column): [ ]
- Missing-data handling: rows with missing values are dropped
- Model: 100-estimator Random Forest classifier
- Any limitations or assumptions: [ ]

Do not include personal or confidential data in screenshots or report attachments.

### 6. Implementation and application screens

Describe the dashboard, model-training form, prediction playground, and per-model monitoring page. Add screenshots and captions, for example:

1. **Model registry:** registered models and summary monitoring metrics.
2. **Model training:** model identifier and dataset upload.
3. **Prediction playground:** example inputs and prediction result.
4. **Monitoring page:** prediction volume, latency, drift status, charts, and recent logs.

### 7. Results

For each model, report observed values from the app:

| Measure | Result |
|---|---|
| Model identifier | [ ] |
| Logged predictions | [ ] |
| Mean latency (ms) | [ ] |
| P95 latency (ms) | [ ] |
| Latency alert (>500 ms) | [Yes/No] |
| Drift status | [Stable/Drift/Insufficient data] |
| Drifted features, if available | [ ] |

State when the values were observed and how many predictions had been submitted. Explain that drift results depend on the available prediction samples and training dataset.

### 8. Testing

List the checks you performed and their outcomes:

- Application starts and dashboard loads: [ ]
- CSV upload and model registration: [ ]
- Prediction submitted and recorded: [ ]
- Monitor page displays latency and drift information: [ ]
- Invalid or insufficient data behavior observed: [ ]

### 9. Limitations and future work

Possible limitations to discuss: no built-in report export, no predictive-quality metrics or train/test evaluation, no automated retraining, and drift detection requires a sufficient number of logged predictions. Only include limitations or future work relevant to your project.

### 10. Conclusion

[Summarize what was implemented, what the monitoring results show, and what could be improved next.]

## Useful project files

- `manage.py` — Django command-line entry point
- `core_project/settings.py` — project settings and SQLite configuration
- `automl/views.py` — training, dashboard, prediction, and monitoring behavior
- `automl/models.py` — registered models and prediction logs
- `automl/drift.py` — data drift checks
- `automl/stats.py` — dataset statistics
- `templates/automl/` — dashboard, training, prediction, and monitoring pages
- `saved_models/` — serialized trained model files
- `saved_datasets/` — uploaded training/reference datasets
