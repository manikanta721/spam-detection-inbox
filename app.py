"""
Flask Web Application for Spam Detector
Based on SMS_Spam_Detection.ipynb approach
Uses TF-IDF + LinearSVM model
"""

import os
import hashlib
import pickle
import re
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import chromadb

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ============================================================
# ChromaDB Setup for User Authentication
# ============================================================
vectordb_path = os.path.join(os.path.dirname(__file__), 'chroma_db')
client = chromadb.PersistentClient(path=vectordb_path)
user_collection = client.get_or_create_collection(name="users")

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    """Create a new user in ChromaDB"""
    try:
        existing = user_collection.get(ids=[username])
        if existing['ids']:
            return False
    except:
        pass
    
    pwd_hash = hash_password(password)
    user_collection.add(
        documents=[username],
        metadatas=[{"username": username, "password_hash": pwd_hash}],
        ids=[username]
    )
    return True

def verify_user(username, password):
    """Verify user credentials"""
    try:
        result = user_collection.get(ids=[username])
        if not result['ids']:
            return False
        stored_hash = result['metadatas'][0]['password_hash']
        return stored_hash == hash_password(password)
    except Exception as e:
        print(f"Error verifying user: {e}")
        return False

# ============================================================
# Model Loading (TF-IDF + LinearSVM)
# ============================================================
print("[INFO] Loading models...")

tfidf_vectorizer = None
svm_model = None
preprocessing_data = None

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

try:
    # Get absolute path to models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Load TF-IDF Vectorizer
    tfidf_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    if os.path.exists(tfidf_path):
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print("✓ TF-IDF vectorizer loaded")
    else:
        print(f"⚠ TF-IDF vectorizer not found at: {tfidf_path}")
    
    # Load SVM Model
    svm_path = os.path.join(models_dir, 'spam_classifier_model.pkl')
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
        print("✓ SVM model loaded")
    else:
        print(f"⚠ SVM model not found at: {svm_path}")
    
    # Load Preprocessing Components
    prep_path = os.path.join(models_dir, 'preprocessing.pkl')
    if os.path.exists(prep_path):
        with open(prep_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
        print("✓ Preprocessing components loaded")
    else:
        print(f"⚠ Preprocessing not found at: {prep_path}")
    
    # Verify all models loaded
    if tfidf_vectorizer is None or svm_model is None or preprocessing_data is None:
        print("⚠ WARNING: Some models failed to load!")
        print("  Please run: python model.py to train models first")
    else:
        print("✅ All models loaded successfully and ready to use!")
    
except Exception as e:
    print(f"⚠ Warning: Could not load models: {e}")
    import traceback
    traceback.print_exc()
    print("  Please run: python model.py to train models first")

# ============================================================
# Prediction Function
# ============================================================
def predict_spam(text):
    """Predict spam using TF-IDF + SVM model"""
    try:
        if tfidf_vectorizer is None or svm_model is None or preprocessing_data is None:
            print("ERROR: Models not loaded in predict_spam")
            return None
        
        # Preprocess text (same as training)
        stemmer = preprocessing_data['stemmer']
        stop_words = preprocessing_data['stop_words']
        cleaned_text = preprocess_text(text, stemmer, stop_words)
        
        # Transform using TF-IDF
        X = tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = svm_model.predict(X)[0]
        
        # Get confidence score using decision_function
        try:
            proba = svm_model.decision_function(X)[0]  # Get confidence score
            # Convert to probability-like score (SVM doesn't have predict_proba)
            # Use decision function distance from hyperplane
            confidence = abs(proba) / (abs(proba) + 1)  # Normalize to 0-1 range
        except:
            # Fallback if decision_function fails
            confidence = 0.5
        
        return {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': float(confidence)
        }
    except Exception as e:
        print(f"ERROR in predict_spam: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    # Debug: Check session
    print(f"[INDEX] Session data: {dict(session)}")
    print(f"[INDEX] Username in session: {'username' in session}")
    if 'username' in session:
        print(f"[INDEX] Redirecting to dashboard (username: {session.get('username')})")
        return redirect(url_for('dashboard'))
    print("[INDEX] Redirecting to login (no session)")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Debug: Check session
    print(f"[LOGIN] Session data: {dict(session)}")
    print(f"[LOGIN] Username in session: {'username' in session}")
    
    # If POST request, handle login
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if verify_user(username, password):
            session['username'] = username
            print(f"[LOGIN] Login successful for: {username}")
            return redirect(url_for('dashboard'))
        else:
            print(f"[LOGIN] Login failed for: {username}")
            return render_template('login.html', error='Invalid username or password')
    
    # GET request - show login page (even if already logged in, allow re-login)
    # If user wants to stay logged in, they can just go to dashboard
    print("[LOGIN] Showing login page")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            return render_template('register.html', error='Please provide both username and password')
        
        if create_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error='Username already exists')
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    # Debug: Check session
    print(f"[DASHBOARD] Session data: {dict(session)}")
    print(f"[DASHBOARD] Username in session: {'username' in session}")
    
    if 'username' not in session:
        print("[DASHBOARD] No username in session, redirecting to login")
        return redirect(url_for('login'))
    
    username = session.get('username')
    print(f"[DASHBOARD] Rendering dashboard for user: {username}")
    return render_template('dashboard.html', username=username)

@app.route('/logout', methods=['POST'])
def logout():
    print(f"[LOGOUT] Logging out user: {session.get('username')}")
    session.pop('username', None)
    session.clear()  # Clear all session data
    print("[LOGOUT] Session cleared, redirecting to login")
    return redirect(url_for('login'))

@app.route('/clear-session', methods=['GET'])
def clear_session():
    """Debug route to clear session - useful for testing"""
    print("[CLEAR-SESSION] Clearing all session data")
    session.clear()
    return redirect(url_for('login'))

@app.route('/force-login', methods=['GET'])
def force_login():
    """Force show login page - clears session first"""
    print("[FORCE-LOGIN] Forcing login page display")
    session.clear()
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for spam prediction"""
    try:
        print("\n[PREDICT] Received prediction request")
        
        if 'username' not in session:
            print("[PREDICT] ERROR: Unauthorized - no session")
            return jsonify({'error': 'Unauthorized'}), 401
        
        print(f"[PREDICT] User: {session.get('username')}")
        
        data = request.get_json()
        if not data:
            print("[PREDICT] ERROR: No JSON data")
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        text = data.get('text', '').strip()
        print(f"[PREDICT] Text received: {text[:50]}...")
        
        if not text:
            print("[PREDICT] ERROR: Empty text")
            return jsonify({'error': 'Text is required'}), 400
        
        # Check if models are loaded
        print(f"[PREDICT] Checking models...")
        print(f"  TF-IDF: {tfidf_vectorizer is not None}")
        print(f"  SVM: {svm_model is not None}")
        print(f"  Preprocessing: {preprocessing_data is not None}")
        
        if tfidf_vectorizer is None or svm_model is None or preprocessing_data is None:
            print("[PREDICT] ERROR: Models not loaded")
            return jsonify({'error': 'Models not loaded. Please train models first.'}), 500
        
        # Get prediction from TF-IDF + SVM model
        print("[PREDICT] Running prediction...")
        result = predict_spam(text)
        
        if result is None:
            print("[PREDICT] ERROR: Prediction returned None")
            return jsonify({'error': 'Prediction failed'}), 500
        
        print(f"[PREDICT] Success! Result: {result}")
        
        return jsonify({
            'classical_prediction': result['prediction'],
            'classical_confidence': result['confidence'],
            'transformer_prediction': None,  # Not using transformer in this version
            'transformer_confidence': None
        })
    
    except Exception as e:
        print(f"[PREDICT] ERROR in /predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Spam Detector - AI-Powered Spam Detection System")
    print("Using TF-IDF + LinearSVM (Based on SMS_Spam_Detection.ipynb)")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open your browser at: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
