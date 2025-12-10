"""
Spam Detection Model Training Script
Based on SMS_Spam_Detection.ipynb approach
Uses TF-IDF + LinearSVM (95.99% accuracy)
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPAM DETECTION MODEL TRAINING")
print("Based on SMS_Spam_Detection.ipynb")
print("=" * 60)

# ============================================================
# 1. Load Dataset
# ============================================================
print("\n[1/5] Loading dataset...")

# Try to use existing dataset
dataset_paths = [
    "../Gen Ai/sms_dataset.csv",
    "../Gen Ai/processed_spam_dataset.csv",
    "sms_dataset.csv"
]

df = None
for path in dataset_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='latin-1')
            print(f"✓ Loaded dataset from: {path}")
            break
        except:
            continue

if df is None:
    print("❌ No dataset found. Please place sms_dataset.csv in gen_ai_pro/")
    exit(1)

# Clean dataset columns
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']].copy()
    df.columns = ['Category', 'Message']
elif 'Category' in df.columns and 'Message' in df.columns:
    df = df[['Category', 'Message']].copy()
elif 'label' in df.columns and 'message' in df.columns:
    df = df.rename(columns={'label': 'Category', 'message': 'Message'})

# Remove empty columns if any
df = df[['Category', 'Message']].copy()
df = df.dropna()

print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Spam samples: {(df['Category'] == 'spam').sum()}, Ham samples: {(df['Category'] == 'ham').sum()}")

# ============================================================
# 2. Balance Dataset (Undersampling - like notebook)
# ============================================================
print("\n[2/5] Balancing dataset (undersampling)...")

minority_len = len(df[df["Category"] == "spam"])
majority_len = len(df[df["Category"] == "ham"])

print(f"  Original: {majority_len} ham, {minority_len} spam")

if majority_len > minority_len:
    # Undersample majority class
    minority_indices = df[df["Category"] == "spam"].index
    majority_indices = df[df["Category"] == "ham"].index
    
    random_majority_indices = np.random.choice(
        majority_indices,
        size=minority_len,
        replace=False
    )
    
    undersampled_indices = np.concatenate([minority_indices, random_majority_indices])
    df = df.loc[undersampled_indices]
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    print(f"  Balanced: {len(df[df['Category'] == 'ham'])}, {len(df[df['Category'] == 'spam'])} spam")

# Create label column
df["Label"] = df["Category"].map({"ham": 0, "spam": 1})

# ============================================================
# 3. Text Preprocessing (exactly like notebook)
# ============================================================
print("\n[3/5] Preprocessing text (stemming + stopwords removal)...")

# Download stopwords if needed
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()
corpus = []

for message in df["Message"]:
    # Remove special characters (keep only letters)
    message = re.sub("[^a-zA-Z]", " ", str(message))
    # Convert to lowercase
    message = message.lower()
    # Split into words
    message = message.split()
    # Stem and remove stopwords
    message = [
        stemmer.stem(word)
        for word in message
        if word not in stop_words
    ]
    # Join back
    message = " ".join(message)
    corpus.append(message)

print(f"✓ Preprocessed {len(corpus)} messages")

# ============================================================
# 4. Train TF-IDF + LinearSVM Model (like notebook)
# ============================================================
print("\n[4/5] Training TF-IDF + LinearSVM Model...")

# TF-IDF Vectorizer (exactly like notebook)
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)  # IMPORTANT: unigrams + bigrams
)

# Transform corpus
X = tfidf.fit_transform(corpus)
y = df['Label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Train LinearSVM (like notebook)
svm_model = LinearSVC(random_state=42, max_iter=2000)
print("  Training SVM model...")
svm_model.fit(X_train, y_train)

# Evaluate Model
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS (TF-IDF + LinearSVM)")
print("=" * 60)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))
print("=" * 60)

# ============================================================
# 5. Save Models
# ============================================================
print("\n[5/5] Saving models...")

os.makedirs('models', exist_ok=True)

# Save TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Saved TF-IDF vectorizer to models/tfidf_vectorizer.pkl")

# Save SVM model
with open('models/spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("✓ Saved SVM model to models/spam_classifier_model.pkl")

# Also save preprocessing function components for app.py
preprocessing_data = {
    'stemmer': stemmer,
    'stop_words': stop_words
}
with open('models/preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing_data, f)
print("✓ Saved preprocessing components to models/preprocessing.pkl")

print("\n✅ Training complete! Models saved to models/")
print("\nNext steps:")
print("  1. Run: python predict.py 'your message here'")
print("  2. Run: python app.py (to start Flask website)")
