import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.features import get_tfidf_vectorizer

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into stratified train and test sets, and vectorizes the text
    features using TF-IDF. Numeric columns are combined with text features using hstack.
    
    Args:
        df (pd.DataFrame): The preprocessed dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (X_train_combined, X_test_combined, y_train, y_test, vectorizer)
    """
    # 1. Stratified Train-Test Split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['fraudulent'], 
        random_state=random_state
    )
    
    y_train = train_df['fraudulent'].values
    y_test = test_df['fraudulent'].values
    
    # 2. Vectorize text features on training set only (prevent data leakage!)
    vectorizer = get_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train_tfidf = vectorizer.fit_transform(train_df['clean_text'].fillna(''))
    X_test_tfidf = vectorizer.transform(test_df['clean_text'].fillna(''))
    
    # 3. Extract binary/numeric features
    numeric_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    X_train_num = csr_matrix(train_df[numeric_cols].values)
    X_test_num = csr_matrix(test_df[numeric_cols].values)
    
    # 4. Horizontally stack text features (TF-IDF) and numeric features
    X_train_combined = hstack([X_train_tfidf, X_train_num])
    X_test_combined = hstack([X_test_tfidf, X_test_num])
    
    return X_train_combined, X_test_combined, y_train, y_test, vectorizer

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Computes standard evaluation metrics: confusion matrix, precision, recall, F1, and ROC-AUC.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'confusion_matrix': cm.tolist()
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
    return metrics
