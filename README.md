# Fake Job Posting Detection System

A production-quality machine learning and NLP system designed to detect fraudulent job advertisements. By combining text preprocessing, TF-IDF vectorization, and metadata analysis, this system predicts if a job posting is legitimate or a scam.

## Features

- **Modular Python Architecture:** Organized pipeline with separated preprocessing, feature extraction, training, and prediction modules.
- **Explainable AI:** Feature coefficients analysis revealing the key terms most associated with fraudulent postings (e.g. *high school*, *data entry*, *earn money*).
- **Interactive Web UI:** Built with Streamlit, enabling users to enter job details and receive real-time authenticity reports with risk score percentages.

---

## Folder Structure

```text
fake job detection/
├── data/                      <- Raw and processed data files (ignored by Git)
├── notebooks/                 <- Jupyter Notebooks detailing step-by-step development
│   ├── 01_env_setup.ipynb
│   ├── 02_dataset_loading.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_advanced_models.ipynb
│   └── 08_evaluation.ipynb
├── src/                       <- Modular production source code
│   ├── __init__.py
│   ├── preprocessing.py       <- Custom regex/NLTK text cleaning utilities
│   ├── features.py            <- TF-IDF configuration
│   ├── train.py               <- Training/Evaluation split helpers
│   └── utils.py               <- Helper functions
├── models/                    <- Saved model pickles (.pkl)
├── app/                       <- Interactive Streamlit application
│   └── app.py
├── requirements.txt           <- Python environment dependencies
├── setup.py                   <- Enables package install of `src`
├── main.py                    <- Root pipeline execution script
└── README.md                  <- Project documentation
```

---

## Installation & Setup

1. **Activate the Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Download Dataset:**
   Place the Kaggle dataset `fake_job_postings.csv` inside the `data/` folder.

---

## How to Run

### Execute the Full Training Pipeline
This cleans the text, fits the TF-IDF Vectorizer, trains a balanced Logistic Regression model, prints evaluation scores, and saves the trained model to `models/`:
```powershell
python main.py
```

### Launch the Streamlit Web Application
```powershell
streamlit run app/app.py
```
This opens the web interface in your browser (usually at `http://localhost:8501`), where you can test live job postings.

---

## Experimental Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | 95.16% | 0.00% | 0.00% | 0.00% | 0.5000 |
| **Naive Bayes** | 96.84% | 84.88% | 42.20% | 56.37% | 0.9382 |
| **Logistic Regression (Balanced)** | 95.86% | 54.33% | **90.75%** | 67.97% | **0.9892** |
| **Linear SVM (Balanced)** | **98.32%** | 81.92% | 83.82% | **82.86%** | 0.9865 |
| **Random Forest (Balanced)** | 96.53% | 60.61% | 80.92% | 69.31% | 0.9843 |
| **XGBoost (Scaled)** | 98.18% | **84.62%** | 76.30% | 80.24% | 0.9780 |

### Key Insights
- **The Imbalance Challenge:** While the Dummy Classifier gets 95.16% accuracy, it detects zero scams. F1-Score is the primary indicator of success.
- **Linear models rule:** Linear SVM and Logistic Regression perform exceptionally well on sparse text spaces, achieving AUC scores near **0.99**.
