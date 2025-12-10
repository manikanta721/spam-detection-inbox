"""
Simple prediction script for spam detection
Based on SMS_Spam_Detection.ipynb approach
Uses TF-IDF + LinearSVM model
"""

import sys
import pickle
import re
import os

def preprocess_text(text, stemmer, stop_words):
    """Preprocess text exactly like in SMS_Spam_Detection.ipynb"""
    # Remove special characters (keep only letters)
    text = re.sub("[^a-zA-Z]", " ", str(text))
    # Convert to lowercase
    text = text.lower()
    # Split into words
    words = text.split()
    # Stem and remove stopwords
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]
    # Join back
    return " ".join(words)

def load_models():
    """Load TF-IDF vectorizer, SVM model, and preprocessing components"""
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/spam_classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/preprocessing.pkl', 'rb') as f:
        preprocessing_data = pickle.load(f)
    return tfidf, model, preprocessing_data

def predict(text, tfidf, model, preprocessing_data):
    """Predict using TF-IDF + SVM model"""
    stemmer = preprocessing_data['stemmer']
    stop_words = preprocessing_data['stop_words']
    
    # Preprocess text
    cleaned = preprocess_text(text, stemmer, stop_words)
    
    # Transform using TF-IDF
    X = tfidf.transform([cleaned])
    
    # Predict
    prediction = model.predict(X)[0]
    proba = model.decision_function(X)[0]
    
    # Convert to confidence score
    confidence = abs(proba) / (abs(proba) + 1)
    
    return {
        'prediction': 'SPAM' if prediction == 1 else 'HAM',
        'confidence': float(confidence)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'your message here'")
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    
    print("=" * 60)
    print("SPAM DETECTION PREDICTION")
    print("Based on SMS_Spam_Detection.ipynb (TF-IDF + LinearSVM)")
    print("=" * 60)
    print(f"\nMessage: {text}\n")
    
    # Check if models exist
    if not os.path.exists('models/tfidf_vectorizer.pkl'):
        print("❌ Models not found. Please run: python model.py")
        sys.exit(1)
    
    # Load models
    print("Loading models...")
    tfidf, model, preprocessing_data = load_models()
    print("✓ Models loaded\n")
    
    # Make prediction
    print("Running prediction...")
    result = predict(text, tfidf, model, preprocessing_data)
    
    # Display results
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("=" * 60)
    
    if result['prediction'] == 'SPAM':
        print("\n⚠️  WARNING: This message is likely SPAM!")
        print("   Be cautious and avoid clicking any links.")
    else:
        print("\n✅ This message appears to be SAFE (HAM)")

if __name__ == '__main__':
    main()
