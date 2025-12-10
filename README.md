# 🛡️ NoJunkZone - Generative AI Spam Detection System

A complete, beginner-friendly spam detection system that combines **Classical ML** (n-grams) and **Generative AI** (DistilBERT) models for accurate spam detection.

![NoJunkZone](https://img.shields.io/badge/NoJunkZone-Spam%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Comparison](#model-comparison)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

NoJunkZone is a production-ready spam detection system that uses two complementary approaches:

1. **Classical ML Model**: Fast, interpretable n-gram based classifier using Logistic Regression
2. **Transformer Model**: State-of-the-art DistilBERT fine-tuned for spam detection

The system includes a beautiful web interface with user authentication, real-time spam scanning, and session statistics.

---

## ✨ Features

- ✅ **Dual Model Architecture**: Classical ML + Transformer AI
- ✅ **Beautiful Web UI**: Modern, responsive design with animations
- ✅ **User Authentication**: Secure login/registration using ChromaDB
- ✅ **Real-time Prediction**: Instant spam detection with confidence scores
- ✅ **Session Statistics**: Track scans, spam detected, and safe messages
- ✅ **Model Comparison**: See predictions from both models side-by-side
- ✅ **Mobile Responsive**: Works perfectly on all devices
- ✅ **Easy to Use**: Simple commands to train, predict, and run

---

## 📁 Project Structure

```
gen_ai_pro/
├── app.py                      # Flask web application
├── model.py                    # Model training script
├── predict.py                  # Command-line prediction script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Saved models (created after training)
│   ├── classical_model.pkl
│   ├── classical_vectorizer.pkl
│   └── transformer_model/     # DistilBERT model files
│
├── chroma_db/                  # ChromaDB user database (created automatically)
│
├── templates/                  # HTML templates
│   ├── login.html
│   ├── register.html
│   └── dashboard.html
│
└── static/                     # CSS and JavaScript
    ├── style.css
    └── script.js
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
cd gen_ai_pro
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installing PyTorch and Transformers may take a few minutes.

### Step 3: Prepare Dataset

Place your SMS spam dataset CSV file in the `gen_ai_pro/` folder. The script will automatically look for:
- `sms_dataset.csv` (with columns: v1=label, v2=message)
- Or any CSV with `label` and `message` columns

If you have the dataset in `../Gen Ai/sms_dataset.csv`, it will be used automatically.

---

## 📖 Usage

### 1. Train the Models

Train both Classical ML and Transformer models:

```bash
python model.py
```

**What happens:**
- Loads and preprocesses the dataset
- Trains Classical ML model (n-grams + Logistic Regression)
- Trains Transformer model (DistilBERT fine-tuning, 3 epochs)
- Saves models to `models/` folder
- Displays performance comparison

**Training Time:**
- Classical Model: ~1-2 minutes
- Transformer Model: ~5-10 minutes (depending on CPU/GPU)

### 2. Command-Line Prediction

Test predictions from command line:

```bash
python predict.py "Congratulations! You've won a free iPhone. Click here now!"
```

**Output:**
```
============================================================
SPAM DETECTION PREDICTION
============================================================

Message: Congratulations! You've won a free iPhone. Click here now!

Loading models...
✓ Models loaded

Running predictions...
============================================================
PREDICTION RESULTS
============================================================

Model                 Prediction     Confidence      
------------------------------------------------------------
Classical (N-gram)    SPAM           0.9876
Transformer (BERT)    SPAM           0.9543
============================================================

✅ Both models agree: SPAM
```

### 3. Run Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

**Features:**
- Register a new account
- Login with your credentials
- Use the dashboard to scan messages
- View real-time statistics

---

## 🧠 How It Works

### Classical ML Model (N-grams)

**How it works:**
1. **Text Preprocessing**: Converts text to lowercase, removes special characters
2. **Feature Extraction**: Uses CountVectorizer with unigrams + bigrams (word pairs)
3. **Model Training**: Logistic Regression classifier
4. **Prediction**: Fast inference based on learned word patterns

**Strengths:**
- Very fast prediction (~5ms)
- Interpretable (can see which words trigger spam)
- Works well with small datasets
- Low memory footprint

**Limitations:**
- Doesn't understand context or word order well
- May miss sophisticated spam patterns

### Transformer Model (DistilBERT)

**How it works:**
1. **Tokenization**: Converts text to token IDs using DistilBERT tokenizer
2. **Model Architecture**: 
   - DistilBERT base model (66M parameters)
   - Fine-tuned on spam dataset
   - Binary classification head
3. **Training**: 3 epochs with early stopping
4. **Prediction**: Context-aware understanding of message meaning

**Strengths:**
- Understands context and word relationships
- Better at detecting sophisticated spam
- Higher accuracy (typically 95-98%)
- Can handle variations in wording

**Limitations:**
- Slower prediction (~50-100ms)
- Requires more computational resources
- Larger model size

### Why Both Models?

- **Classical Model**: Fast, reliable baseline
- **Transformer Model**: Advanced, context-aware detection
- **Combined**: Better accuracy and reliability through agreement checking

---

## 📊 Model Comparison

After training, you'll see a comparison like this:

```
============================================================
MODEL COMPARISON SUMMARY
============================================================
Metric           Classical       Transformer    
------------------------------------------------------------
Accuracy         0.9234          0.9678
Precision        0.9123          0.9543
Recall           0.9456          0.9789
F1-Score         0.9287          0.9665
============================================================
```

**Typical Performance:**
- **Classical Model**: 90-94% accuracy
- **Transformer Model**: 95-98% accuracy

---

## 🌐 API Endpoints

### Web Routes

- `GET /` - Redirects to login or dashboard
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /register` - Registration page
- `POST /register` - Create new user
- `GET /dashboard` - Main dashboard (requires login)
- `POST /logout` - Logout user

### Prediction API

**Endpoint**: `POST /predict`

**Request:**
```json
{
    "text": "Your message here"
}
```

**Response:**
```json
{
    "classical_prediction": "SPAM",
    "classical_confidence": 0.9876,
    "transformer_prediction": "SPAM",
    "transformer_confidence": 0.9543
}
```

**Authentication**: Requires valid session (logged in user)

---

## 🗄️ ChromaDB User Database

NoJunkZone uses **ChromaDB** (vector database) for user authentication:

- **Location**: `chroma_db/` folder (created automatically)
- **Storage**: Usernames and password hashes
- **Security**: Passwords are hashed using SHA-256

**How it works:**
- Each user is stored with:
  - `id`: username
  - `metadata`: `{username, password_hash}`
  - `document`: username string

---

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **scikit-learn**: Classical ML models
- **Transformers (HuggingFace)**: DistilBERT model
- **PyTorch**: Deep learning framework
- **ChromaDB**: Vector database for user storage
- **pandas**: Data processing
- **numpy**: Numerical operations

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations
- **JavaScript (ES6+)**: Frontend interactions
- **No frameworks**: Pure vanilla JS for simplicity

---

## 📝 Example Usage

### Training Example

```bash
$ python model.py

[1/6] Loading dataset...
✓ Loaded dataset from: ../Gen Ai/sms_dataset.csv
✓ Dataset shape: (5572, 2)
✓ Spam samples: 747, Ham samples: 4825

[2/6] Preparing data...
✓ Train: 4457, Test: 1115

[3/6] Training Classical ML Model...
Accuracy:  0.9234
Precision: 0.9123
Recall:    0.9456
F1-Score:  0.9287

[5/6] Training Transformer Model...
[Training progress...]

✅ Training complete!
```

### Web Application Example

1. **Register**: Create account with username/password
2. **Login**: Authenticate with credentials
3. **Scan**: Paste message → Click "Scan for Spam"
4. **Results**: See predictions from both models
5. **Stats**: View session statistics

---

## 🎨 UI Features

- **Glassmorphism Design**: Modern frosted glass effect
- **Gradient Animations**: Smooth background gradients
- **Floating Cards**: Subtle card animations
- **Responsive Layout**: Works on mobile, tablet, desktop
- **Loading States**: Visual feedback during predictions
- **Color-coded Results**: Green (HAM) / Red (SPAM)

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: "Model not found"
**Solution**: Train models first: `python model.py`

### Issue: "Dataset not found"
**Solution**: Place `sms_dataset.csv` in `gen_ai_pro/` folder

### Issue: Training is slow
**Solution**: Normal for transformer model. Use GPU if available for faster training.

---

## 📚 Learning Resources

### Understanding the Models

1. **Classical ML**: Learn about n-grams and Logistic Regression
2. **Transformers**: Study BERT architecture and fine-tuning
3. **ChromaDB**: Vector database concepts

### Extending the Project

- Add more models (SVM, Random Forest)
- Implement model ensemble voting
- Add email spam detection
- Create API documentation
- Add unit tests

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- **HuggingFace**: For DistilBERT model
- **scikit-learn**: For classical ML tools
- **ChromaDB**: For vector database

---

## 📞 Support

For questions or issues:
1. Check the README
2. Review error messages
3. Ensure all dependencies are installed
4. Verify dataset format

---

**Built with ❤️ for spam-free communication**

*NoJunkZone - Your AI-powered spam shield*

