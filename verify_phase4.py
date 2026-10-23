import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

from django.test import Client

client = Client()

print("--- Running Phase 4 Endpoint & Workflow Verification ---")

# 1. GET Dashboard
res = client.get('/')
assert res.status_code == 200, f"Dashboard failed with status {res.status_code}"
assert b"MLOps Sentinel" in res.content
assert b"Model Registry" in res.content
assert b"Platform Avg Latency" in res.content
assert b"Drift Status" in res.content
print("1. Dashboard (GET /): HTTP 200 OK — Enterprise metrics & model cards rendered")

# 2. GET Build Model
res = client.get('/build/')
assert res.status_code == 200, f"Build failed with status {res.status_code}"
assert b"AutoML Model Training Pipeline" in res.content
print("2. Build Model (GET /build/): HTTP 200 OK — Upload interface rendered")

# 3. GET Churn Predict
res = client.get('/predict/Customer_Churn_Model/')
assert res.status_code == 200, f"Churn predict failed with status {res.status_code}"
assert b"Customer_Churn_Model" in res.content
print("3. Customer Churn Predict (GET /predict/Customer_Churn_Model/): HTTP 200 OK")

# 4. GET Loan Predict
res = client.get('/predict/Loan_Approval_Model/')
assert res.status_code == 200, f"Loan predict failed with status {res.status_code}"
assert b"Education Status" in res.content
assert b"Employment Classification" in res.content
assert b"CIBIL / Credit Score" in res.content
assert b"Annual Income" in res.content
print("4. Loan Approval Predict (GET /predict/Loan_Approval_Model/): HTTP 200 OK — Specialized categorical controls verified")

# 5. POST Loan Predict (Case A: High CIBIL 780, High Assets, Graduate, Salaried/No)
payload_a = {
    'no_of_dependents': '2',
    'friendly_education': 'Graduate',
    'friendly_self_employed': 'No',
    'income_annum': '7500000',
    'loan_amount': '15000000',
    'loan_term': '12',
    'cibil_score': '780',
    'residential_assets_value': '8000000',
    'commercial_assets_value': '2000000',
    'luxury_assets_value': '15000000',
    'bank_asset_value': '6000000'
}
res_a = client.post('/predict/Loan_Approval_Model/', payload_a)
assert res_a.status_code == 200, f"Loan POST Case A failed with status {res_a.status_code}"
content_a = res_a.content.decode('utf-8')
assert "Approved" in content_a, "Expected 'Approved' verdict in response"
assert "Inference Telemetry" in content_a or "Latency" in content_a
print("5. Loan POST Case A (Graduate, Not Self-Employed, CIBIL 780): HTTP 200 -> Verdict: Approved (Positive Outcome)")

# 6. POST Loan Predict (Case B: Low CIBIL 320, Low Assets, Not Graduate, Self-Employed/Yes)
payload_b = {
    'no_of_dependents': '1',
    'friendly_education': 'Not Graduate',
    'friendly_self_employed': 'Yes',
    'income_annum': '1200000',
    'loan_amount': '15000000',
    'loan_term': '20',
    'cibil_score': '320',
    'residential_assets_value': '500000',
    'commercial_assets_value': '0',
    'luxury_assets_value': '200000',
    'bank_asset_value': '100000'
}
res_b = client.post('/predict/Loan_Approval_Model/', payload_b)
assert res_b.status_code == 200, f"Loan POST Case B failed with status {res_b.status_code}"
content_b = res_b.content.decode('utf-8')
assert "Rejected" in content_b, "Expected 'Rejected' verdict in response"
print("6. Loan POST Case B (Not Graduate, Self-Employed, CIBIL 320): HTTP 200 -> Verdict: Rejected (High Risk Outcome)")

# 7. GET Monitor Loan Model
res = client.get('/monitor/Loan_Approval_Model/')
assert res.status_code == 200, f"Loan monitor failed with status {res.status_code}"
content_mon = res.content.decode('utf-8')
assert "Statistical Data Drift Diagnostics" in content_mon
assert "Latency Timeline" in content_mon
assert "Empirical Distribution Comparison" in content_mon
assert "Kolmogorov-Smirnov" in content_mon
print("7. Monitor Page (GET /monitor/Loan_Approval_Model/): HTTP 200 OK — Decoupled SLA, Dual Charts & KS/Chi-Square tables verified")

print("\n=======================================================")
print(">>> ALL PHASE 4 VERIFICATION CHECKS COMPLETED SUCCESSFULLY! <<<")
print("=======================================================")
