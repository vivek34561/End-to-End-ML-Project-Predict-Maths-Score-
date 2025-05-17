# 📊 Student Performance ML Project

This project focuses on building a machine learning pipeline to analyze and predict student performance using modern tools like MLflow, DVC, and CatBoost.

---

## 🚀 Features

- Full ML pipeline from ingestion to prediction
- Modular component-based architecture
- Experiment tracking with MLflow
- Data and model versioning with DVC
- Logging and custom error handling

---

## 🏗️ Project Structure

ML_Project/
│
├── .dvc/                        # DVC metadata and pipeline files
├── .gitignore                  # Git ignore configuration
├── .dvcignore                  # DVC ignore rules
├── .env                        # Environment variables
├── Dockerfile                  # Docker configuration for reproducible environment
├── README.md                   # Project documentation
├── app.py                      # Main application script (e.g. FastAPI/Flask entry)
├── requirements.txt            # List of dependencies
├── setup.py                    # Package configuration
├── setup.pymain.py             # Possibly entry script for setup
├── template.py                 # Template/boilerplate script
│
├── artifact/                   # Intermediate processed data
├── artifacts/                  # Output files from experiments or models
├── catboost_info/             # CatBoost training logs/info
├── logs/                       # Log files
├── mlruns/                     # MLFlow tracking directory
│
├── notebook/
│   └── data/
│       ├── raw.csv
│       ├── EDA_student_performance.ipynb
│       └── MODEL_TRANING.ipynb
│
└── src/
    └── mlproject/
        ├── __init__.py
        ├── exception.py              # Custom exception class
        ├── logger.py                 # Logger config
        ├── utils.py                  # Utility functions
        │
        ├── components/               # Modular ML components
        │   ├── __init__.py
        │   ├── data_ingestion.py
        │   ├── data_transformation.py
        │   ├── model_trainer.py
        │   └── model_monitoring.py
        │
        ├── pipelines/                # ML pipelines
        │   ├── __init__.py
        │   ├── training_pipelines.py
        │   └── prediction_pipeline.py



### Download Project_structure_pdf 📁 [ML_Project_Structure.pdf](https://github.com/user-attachments/files/20263587/ML_Project_Structure.pdf)

See the project tree above ☝️ for details.

--- 

## 🧱 Major Components

| Module               | Description |
|----------------------|-------------|
| `components/`        | Contains modular scripts for ingestion, transformation, training, monitoring |
| `pipelines/`         | Training and prediction orchestration |
| `notebook/`          | EDA and prototyping |
| `src/mlproject/`     | Core source files and utilities |
| `app.py`             | Entry point  |
| `.dvc`, `mlruns`     | DVC and MLflow tracking |

---

## ⚙️ Setup Instructions

### 🐍 Create Environment

```bash
conda create -n ml_project python=3.10 -y
conda activate ml_project
````

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 Run App

```bash
python app.py
```

---

## 🧪 Experiment Tracking with MLflow

```bash
mlflow ui
# Access via http://localhost:5000
```

---

## 💾 Data Versioning with DVC

```bash
dvc init
dvc add path/to/data.csv
git add .gitignore data.csv.dvc
```



---

## 📂 Logs

Logs are saved under the `/logs` directory.

---

## 📧 Contact

Maintained by Vivek Kumar Gupta . Feel free to contribute or raise an issue.

For any inquiries, please reach out via email or GitHub.


---

```

