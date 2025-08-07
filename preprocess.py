import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('news.csv')  # Replace with your file path

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Combine title + text for better context
df['combined_text'] = df['title'] + " " + df['text']

# Clean the combined text
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Map labels to binary (0=REAL, 1=FAKE)
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# Save cleaned data
df.to_csv('cleaned_news_dataset.csv', index=False)