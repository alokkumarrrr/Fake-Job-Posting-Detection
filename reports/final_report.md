# Project Report: Fake Job Posting Detection System Using Machine Learning and NLP

**Author:** Computer Science Department Student  
**Task:** Binary Text Classification (Legitimate vs. Fraudulent)  
**Dataset:** Real/Fake Job Posting Prediction Dataset (Kaggle)

---

## 1. Abstract
Online recruitment fraud has grown rapidly, leading to financial loss, identity theft, and personal information compromise. This report details the construction of a production-grade machine learning system designed to automatically detect fake job postings. Using 17,880 postings, we apply NLP text cleaning, TF-IDF vectorization, stack tabular metadata, and benchmark several algorithms (Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost). Our champion model (Linear SVM) achieves an F1-score of **82.86%** and an ROC-AUC of **0.986**, and is deployed via an interactive Streamlit web interface.

---

## 2. Introduction & Problem Statement
Job search engines are heavily targeted by scammers who post fraudulent positions to acquire personal identifiable information (PII), charge application fees, or commit phishing. The goal is to build an automated classification pipeline.
The dataset target is binary:
- **0 → Legitimate Posting**
- **1 → Fraudulent Posting**

A key technical challenge is the severe **class imbalance**—only 4.84% of job postings in the dataset are fraudulent. Standard evaluation metrics like accuracy fail in this scenario, requiring the optimization of Precision, Recall, and F1-score.

---

## 3. Exploratory Data Analysis (EDA) & Insights
Our data analysis revealed key features that differentiate fake jobs:
1. **The Company Logo Effect:** 
   - Postings **without** a company logo have a **15.93%** fraud rate.
   - Postings **with** a logo have a mere **1.99%** fraud rate.
   - A missing company logo makes a posting roughly **8 times** more likely to be a scam.
2. **Screening Questions:** 
   - Scammers include screening questions in only **2.84%** of cases, compared to legitimate postings which include questions in **6.78%** of cases. Scammers prefer low-friction applications.
3. **Text Length Discrepancy:**
   - **Legitimate** company profiles average **640 characters**.
   - **Fraudulent** company profiles average only **230 characters**. Scammers write brief, generic descriptions.

---

## 4. Text Preprocessing Pipeline
Text fields (`title`, `company_profile`, `description`, `requirements`, `benefits`) were combined into a unified text document and passed to our modular preprocessing script (`src/preprocessing.py`). The pipeline performs:
1. **Lowercasing:** Converts text to lower case for uniformity.
2. **HTML Stripping:** Removes remnants of HTML formatting (e.g. `<br>`).
3. **Punctuation & Number Removal:** Retains only alphabetic text.
4. **Stopword Filtering:** Eliminates high-frequency grammatical connector words (e.g., *and*, *the*, *is*) using the NLTK English stopword database.
5. **Lemmatization:** Uses NLTK's `WordNetLemmatizer` to reduce tokens to their base root dictionary forms (e.g., *requirements* -> *requirement*).

---

## 5. Feature Engineering
We converted the cleaned text into numbers using **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency) config:
- `max_features = 5000` (limiting vocabulary to the top 5,000 terms to avoid overfitting).
- `ngram_range = (1, 2)` (capturing both single words and two-word phrases).

The TF-IDF sparse matrix was stacked horizontally with three tabular binary features (`telecommuting`, `has_company_logo`, `has_questions`) using `scipy.sparse.hstack` for training.

---

## 6. Experimental Results & Benchmarking

We partitioned the dataset using a **Stratified 80/20 Train/Test Split** to prevent data leakage and maintain target class distribution.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | 95.16% | 0.00% | 0.00% | 0.00% | 0.5000 |
| **Naive Bayes** | 96.84% | **84.88%** | 42.20% | 56.37% | 0.9382 |
| **Logistic Regression (Balanced)** | 95.86% | 54.33% | **90.75%** | 67.97% | **0.9892** |
| **Linear SVM (Balanced)** | **98.32%** | 81.92% | 83.82% | **82.86%** | 0.9865 |
| **Random Forest (Balanced)** | 96.53% | 60.61% | 80.92% | 69.31% | 0.9843 |
| **XGBoost (Scaled)** | 98.18% | 84.62% | 76.30% | 80.24% | 0.9780 |

### Key Observations:
- **Logistic Regression** is optimal for high-recall tasks (catching 90.75% of scams) and exhibits the highest overall ROC-AUC (**0.9892**).
- **Linear SVM** acts as the balanced champion model, yielding an F1-score of **82.86%** and making only 32 false positive classifications.

---

## 7. Model Interpretability (Explainable AI)
Extracting the coefficients of the Logistic Regression model reveals features with the highest weight in classifying job authenticity:

### Top Fraud Indicators (Positive Weights)
1. **"high school"** (Weight: 3.59) - targeting low-skill applicant bases.
2. **"aptitude"** (Weight: 3.58) - generic screening phrases.
3. **"link"** (Weight: 3.49) - directs victims to external credential-harvesting forms.
4. **"earn" / "money"** (Weight: 3.28 / 2.98) - quick financial gains.
5. **"data entry" / "receptionist" / "administrative assistant"** - highly targeted industries.

### Top Legitimacy Indicators (Negative Weights)
1. **"english"** (Weight: -2.63)
2. **"growing"** (Weight: -2.34)
3. **"has_company_logo"** (Weight: -2.18)
4. **"startup"** (Weight: -1.90)

---

## 8. Conclusion
We successfully designed, trained, evaluated, and deployed a production-grade Fake Job Detection system. Linear models are highly effective at generalizing text representations compared to tree-based ensembles in sparse high-dimensional matrices. The Streamlit deployment exposes this model via an intuitive, real-world interface.
