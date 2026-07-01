import os
import joblib
import pandas as pd
from src.preprocessing import clean_text
from src.train import prepare_data, evaluate_predictions
from sklearn.linear_model import LogisticRegression

def run_pipeline():
    print("=== Starting Fake Job Posting Detection Pipeline ===")
    
    # 1. Load raw dataset
    raw_path = os.path.join("data", "fake_job_postings.csv")
    if not os.path.exists(raw_path):
        print(f"Error: Raw dataset not found at {raw_path}. Please place it in data/ folder first.")
        return
        
    print("Loading raw dataset...")
    df = pd.read_csv(raw_path)
    
    # 2. Combine and clean text fields
    print("Combining and preprocessing text fields...")
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('')
    
    df['combined_text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']
    df['clean_text'] = df['combined_text'].apply(clean_text)
    
    # Keep key columns
    processed_df = df[['clean_text', 'telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']]
    processed_df = processed_df[processed_df['clean_text'].str.strip() != '']
    
    # Save processed CSV for quick reference
    os.makedirs("data", exist_ok=True)
    processed_csv_path = os.path.join("data", "processed_jobs.csv")
    processed_df.to_csv(processed_csv_path, index=False)
    print(f"Saved processed dataset to {processed_csv_path}")
    
    # 3. Prepare features and split
    print("Vectorizing and splitting data...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(processed_df)
    
    # 4. Train Model
    print("Training Logistic Regression Model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    
    print("\n=== Model Performance Evaluation ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    
    # 6. Save Model Artifacts
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "logistic_regression.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\nSaved trained model to: {model_path}")
    print(f"Saved TF-IDF Vectorizer to: {vectorizer_path}")
    print("=== Pipeline Run Finished Successfully! ===")

if __name__ == "__main__":
    run_pipeline()
