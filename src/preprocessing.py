import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pre-download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text: str) -> str:
    """
    Cleans raw text by performing the following operations:
    1. Lowercasing the text.
    2. Removing HTML tags.
    3. Removing non-alphabetic characters (numbers, punctuation).
    4. Tokenizing and removing English stop words.
    5. Lemmatizing tokens to their base form.
    
    Args:
        text (str): The raw input text.
        
    Returns:
        str: The preprocessed and cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Keep only alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 4. Tokenization (split by whitespace)
    words = text.split()
    
    # 5. Stopword removal and Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]
    
    return " ".join(cleaned_words)
