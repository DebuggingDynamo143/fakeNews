# train.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

def clean_text(text):
    """Text cleaning function matching app.py"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data():
    """Load and validate dataset"""
    try:
        df = pd.read_csv('cleaned_news_dataset.csv')
        
        print("\nData Quality Check:")
        print("Missing values per column:")
        print(df.isnull().sum())
        
        # Convert label column to string if it isn't already
        df['label'] = df['label'].astype(str)
        
        # Clean label values
        df['label'] = df['label'].str.lower().str.strip()
        
        # Filter only valid labels
        valid_labels = ['real', 'fake', '0', '1']
        df = df[df['label'].isin(valid_labels)]
        
        if len(df) == 0:
            raise ValueError("No valid labels found (must be 'real', 'fake', 0, or 1)")
            
        return df
    
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("\nPlease ensure:")
        print("1. 'cleaned_news_dataset.csv' exists")
        print("2. Label column contains only 'real', 'fake', 0, or 1")
        exit()

# Main execution
if __name__ == '__main__':
    print("Starting model training...")
    df = load_data()
    
    # Preprocess
    print("\nPreprocessing text...")
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Convert labels to numeric (0=real, 1=fake)
    df['label'] = df['label'].replace({
        'real': 0,
        'fake': 1,
        '0': 0,
        '1': 1
    }).astype(int)

    # Vectorize
    print("\nVectorizing text...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        ngram_range=(1, 2)
    )
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label'].values

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    print("\nTraining model...")
    model = LogisticRegression(
        max_iter=1000,
        C=0.5,
        solver='liblinear'
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['real', 'fake']))

    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    print("\nModel successfully saved to:")
    print("- model.pkl")
    print("- vectorizer.pkl")