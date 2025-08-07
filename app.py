# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import re
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model components
tfidf_vectorizer = None
model = None

def clean_text(text):
    """Enhanced text cleaning"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def load_model():
    """Load the trained model and vectorizer"""
    global tfidf_vectorizer, model
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print("Trained model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure model.pkl and vectorizer.pkl exist in the directory")
        print("Run train.py first if you haven't")
        exit()

load_model()

@app.route('/')
def home():
    """Render the main page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fake News Detector</title>
        <style>
            * {
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            body {
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background-color: #2c3e50;
                color: white;
                padding: 20px 0;
                text-align: center;
                margin-bottom: 30px;
            }
            h1 {
                margin: 0;
                font-size: 2.5rem;
            }
            .description {
                max-width: 800px;
                margin: 20px auto;
                text-align: center;
                color: #7f8c8d;
            }
            .detection-area {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #2c3e50;
            }
            textarea, input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
                resize: vertical;
                min-height: 150px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                display: none;
            }
            .real {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .fake {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .file-upload {
                margin-top: 20px;
            }
            .file-upload input {
                margin-bottom: 10px;
            }
            footer {
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                color: #7f8c8d;
                font-size: 14px;
            }
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                h1 {
                    font-size: 1.8rem;
                }
                .detection-area {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="container">
                <h1>Fake News Detector</h1>
            </div>
        </header>
        
        <div class="container">
            <div class="description">
                <p>Detect potentially fake news articles by entering text or uploading a file. Our AI analyzes the content for signs of misinformation.</p>
            </div>
            
            <div class="detection-area">
                <div class="input-group">
                    <label for="news-title">News Title (optional):</label>
                    <input type="text" id="news-title" placeholder="Enter news headline...">
                </div>
                
                <div class="input-group">
                    <label for="news-text">News Content:</label>
                    <textarea id="news-text" placeholder="Paste the news article content here..."></textarea>
                </div>
                
                <div class="input-group">
                    <label for="news-url">Or Enter News URL (optional):</label>
                    <input type="text" id="news-url" placeholder="https://example.com/news-article">
                </div>
                
                <div class="file-upload">
                    <label>Or Upload a File:</label>
                    <input type="file" id="file-input" accept=".txt,.pdf,.docx">
                    <small>Supported formats: TXT, PDF, DOCX (PDF/DOCX processing is simulated)</small>
                </div>
                
                <button id="check-btn">Check for Fake News</button>
                
                <div id="result" class="result"></div>
                <div id="details" style="margin-top: 20px;"></div>
            </div>
        </div>
        
        <footer>
            <div class="container">
                <p>© 2023 Fake News Detector | For demonstration purposes only</p>
            </div>
        </footer>
        
        <script>
            document.getElementById('check-btn').addEventListener('click', async function() {
                const title = document.getElementById('news-title').value;
                const text = document.getElementById('news-text').value;
                const url = document.getElementById('news-url').value;
                const fileInput = document.getElementById('file-input');
                
                let content = text;
                let usedTitle = title;
                
                // If URL is provided but no text, simulate fetching
                if (!text && url) {
                    usedTitle = `Content from ${url}`;
                    content = `This would be fetched content from ${url}. In a real app, we would download and extract the article text.`;
                }
                
                // If file is uploaded
                if (fileInput.files.length > 0 && !text && !url) {
                    const file = fileInput.files[0];
                    const fileName = file.name.toLowerCase();
                    
                    if (fileName.endsWith('.txt')) {
                        content = await readTextFile(file);
                    } else if (fileName.endsWith('.pdf') || fileName.endsWith('.docx')) {
                        content = `Simulated content from ${file.name}`;
                    }
                    usedTitle = `Uploaded file: ${file.name}`;
                }
                
                if (!content) {
                    alert('Please enter some text, a URL, or upload a file.');
                    return;
                }
                
                const response = await fetch('/check', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        title: usedTitle,
                        text: content 
                    }),
                });
                
                const data = await response.json();
                showResult(data);
            });
            
            function readTextFile(file) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = event => resolve(event.target.result);
                    reader.onerror = error => reject(error);
                    reader.readAsText(file);
                });
            }
            
            function showResult(data) {
                const resultDiv = document.getElementById('result');
                const detailsDiv = document.getElementById('details');
                
                resultDiv.style.display = 'block';
                
                if (data.error) {
                    resultDiv.textContent = `Error: ${data.error}`;
                    resultDiv.className = 'result fake';
                    detailsDiv.innerHTML = '';
                    return;
                }
                
                resultDiv.textContent = `Result: ${data.result.toUpperCase()} (${data.confidence}% confidence)`;
                resultDiv.className = 'result';
                resultDiv.classList.add(data.result === 'real' ? 'real' : 'fake');
                
                // Show additional details
                detailsDiv.innerHTML = `
                    <h3>Analysis Details:</h3>
                    <p><strong>Processed Text:</strong> ${data.processed_text.substring(0, 200)}...</p>
                `;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/check', methods=['POST'])
def check_news():
    """Check if news is fake"""
    try:
        data = request.get_json()
        title = data.get('title', '')
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Combine title and text if title exists
        combined_text = f"{title} {text}" if title else text
        
        # Clean the text
        cleaned_text = clean_text(combined_text)
        
        # Vectorize the text
        tfidf_text = tfidf_vectorizer.transform([cleaned_text])
        
        # Make prediction (assuming model returns 0=real, 1=fake)
        prediction_num = model.predict(tfidf_text)[0]
        prediction = 'fake' if prediction_num == 1 else 'real'
        
        # Get confidence score using decision function
        decision_score = model.decision_function(tfidf_text)[0]
        confidence = int(np.clip(decision_score * 10 + 50, 0, 100))  # Convert to 0-100% scale
        
        return jsonify({
            'result': prediction,
            'confidence': confidence,
            'processed_text': cleaned_text
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'result': 'error',
            'confidence': 0
        }), 500

if __name__ == '__main__':
    app.run(debug=True)