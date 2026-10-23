pipeline {
    agent any

    environment {
        PYTHONUNBUFFERED = '1'
        PYTHONDONTWRITEBYTECODE = '1'
    }

    stages {
        stage('Checkout & Diagnostics') {
            steps {
                checkout scm
                echo '==================================================='
                echo ' Stage 1: Checkout & Environment Diagnostics'
                echo '==================================================='
                bat 'echo Host Operating System: %OS%'
                bat 'git --version'
                bat 'docker --version || echo [WARNING] Docker CLI not accessible in Jenkins service PATH'
            }
        }

        stage('Dependency Preparation') {
            steps {
                echo '==================================================='
                echo ' Stage 2: Environment Setup & Dependency Audit'
                echo '==================================================='
                bat '''
                    if exist venv\\Scripts\\python.exe (
                        echo Found existing virtual environment. Verifying packages...
                        venv\\Scripts\\python.exe -m pip install -r requirements.txt --quiet
                    ) else (
                        echo Setting up CI virtual environment...
                        python -m venv venv
                        venv\\Scripts\\python.exe -m pip install --upgrade pip
                        venv\\Scripts\\python.exe -m pip install -r requirements.txt
                    )
                '''
            }
        }

        stage('Django System Check') {
            steps {
                echo '==================================================='
                echo ' Stage 3: Django System & Configuration Integrity'
                echo '==================================================='
                bat '''
                    if exist venv\\Scripts\\python.exe (
                        venv\\Scripts\\python.exe manage.py check --fail-level WARNING
                    ) else (
                        python manage.py check --fail-level WARNING
                    )
                '''
            }
        }

        stage('Automated Test Suite') {
            steps {
                echo '==================================================='
                echo ' Stage 4: Automated Pytest Suite & MLOps Tests'
                echo '==================================================='
                bat '''
                    if not exist reports mkdir reports
                    if exist venv\\Scripts\\pytest.exe (
                        venv\\Scripts\\pytest.exe -v --junitxml=reports/pytest-results.xml
                    ) else (
                        pytest -v --junitxml=reports/pytest-results.xml
                    )
                '''
            }
        }

        stage('Statistical Drift Verification') {
            steps {
                echo '==================================================='
                echo ' Stage 5: Reproducible Statistical Drift Pipeline'
                echo '==================================================='
                bat '''
                    if exist venv\\Scripts\\python.exe (
                        set PY_CMD=venv\\Scripts\\python.exe
                    ) else (
                        set PY_CMD=python
                    )
                    echo [1/3] Simulating normal traffic baseline (35 samples)...
                    %PY_CMD% manage.py simulate_drift --model-name Loan_Approval_Model --mode normal --count 35 --clear-existing
                    %PY_CMD% -c "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings'); django.setup(); from automl.models import TrainedModel; from automl.drift import detect_data_drift; m=TrainedModel.objects.get(model_name='Loan_Approval_Model'); r=detect_data_drift(m, min_samples=25); assert not r['drift_detected'], 'ERROR: Normal mode unexpectedly flagged statistical drift!'"

                    echo [2/3] Simulating deliberate macroeconomic covariate shift (35 samples)...
                    %PY_CMD% manage.py simulate_drift --model-name Loan_Approval_Model --mode drifted --count 35 --clear-existing
                    %PY_CMD% -c "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings'); django.setup(); from automl.models import TrainedModel; from automl.drift import detect_data_drift; m=TrainedModel.objects.get(model_name='Loan_Approval_Model'); r=detect_data_drift(m, min_samples=25); assert r['drift_detected'], 'ERROR: Deliberate shift was not flagged as statistical drift!'"

                    echo [3/3] Restoring stable baseline traffic for model...
                    %PY_CMD% manage.py simulate_drift --model-name Loan_Approval_Model --mode normal --count 35 --clear-existing
                '''
            }
        }

        stage('Docker Image Build & Test') {
            steps {
                echo '==================================================='
                echo ' Stage 6: Multi-Stage Docker Image Build & Verification'
                echo '==================================================='
                bat '''
                    docker build -t model-monitoring-dashboard:latest .
                    echo Validating container startup and system check...
                    docker run --rm model-monitoring-dashboard:latest python manage.py check
                '''
            }
        }
    }

    post {
        always {
            echo '==================================================='
            echo ' Stage 7: Post-Build Reporting & Artifact Archiving'
            echo '==================================================='
            junit allowEmptyResults: true, testResults: 'reports/pytest-results.xml'
        }
        success {
            echo '>>> CI/CD PIPELINE SUCCEEDED: All MLOps Quality Gates Passed! <<<'
        }
        failure {
            echo '>>> CI/CD PIPELINE FAILED: One or more stages failed. Check console output above. <<<'
        }
    }
}
