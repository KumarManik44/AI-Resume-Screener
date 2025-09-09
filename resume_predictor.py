
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
def ensure_nltk_data():
    """Download NLTK data if not present"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Initialize preprocessing tools
ensure_nltk_data()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean resume text - FIXED regex patterns"""
    if pd.isna(text) or text == '' or len(str(text).strip()) == 0:
        return ""

    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)  # Remove phone numbers
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '', text)  # Remove phone numbers
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n+', ' ', text)  # Newlines to spaces
    text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    text = ' '.join(text.split())  # Clean extra whitespace
    return text.strip()

def advanced_preprocess(text):
    """Advanced preprocessing with tokenization and lemmatization"""
    if not text or text == "":
        return ""

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def predict_resume_category(resume_text, model_path='resume_screener_model.pkl'):
    """Predict job category for resume text"""
    # Load model package
    model_package = joblib.load(model_path)

    model = model_package['model']
    vectorizer = model_package['vectorizer']
    label_encoder = model_package['label_encoder']

    # Preprocess using same pipeline as training
    cleaned_text = clean_text(resume_text)
    processed_text = advanced_preprocess(cleaned_text)

    # Vectorize and predict
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    prediction_proba = model.predict_proba(text_vector)[0]

    # Format results
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    confidence = prediction_proba[prediction]

    top_3_indices = prediction_proba.argsort()[-3:][::-1]
    top_3_categories = label_encoder.inverse_transform(top_3_indices)
    top_3_probabilities = prediction_proba[top_3_indices]

    return {
        'predicted_category': predicted_category,
        'confidence': float(confidence),
        'top_3_predictions': [
            {'category': cat, 'probability': float(prob)}
            for cat, prob in zip(top_3_categories, top_3_probabilities)
        ]
    }

if __name__ == "__main__":
    # Example usage
    sample_resume = "Software engineer with 5 years experience in Python, machine learning, and data science..."
    result = predict_resume_category(sample_resume)
    print("Prediction:", result)
