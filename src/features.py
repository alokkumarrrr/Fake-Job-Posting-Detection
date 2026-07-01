from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features: int = 5000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Returns a configured Scikit-Learn TfidfVectorizer instance.
    
    Args:
        max_features (int): The maximum vocabulary size (top N most frequent terms).
        ngram_range (tuple): The minimum and maximum limits for n-grams.
        
    Returns:
        TfidfVectorizer: A configured vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True  # Scales term frequencies logarithmically to reduce the impact of very high counts
    )
