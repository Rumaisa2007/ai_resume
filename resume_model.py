import os
import re
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab',quiet=True)


SKILL_KEYWORDS = {
    'python', 'java','c++', 'sql', 'machine learning', 'deep learning',
    'aws', 'docker', 'spark', 'hadoop', 'javascript', 'react', 'node',
    'data analysis', 'nlp', 'computer vision', 'pandas', 'numpy', 'tensorflow'
}
PROJECT_KEYWORDS = {
    'recommendation', 'web app', 'mobile app', 'data pipeline',
    'database', 'cloud', 'api', 'automation', 'security', 'devops'
}


def extract_text_from_pdf(pdf_path):
## to extract words from pdf.   
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.lower()

def clean_text(text):
##  to clean and tokenize text.
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def extract_experience(text):
##  to extract years of experience using regex.  
    exp_pattern = r'(\d+\.?\d*)\s*(?:years?|yrs?)\s*of?\s*experience'
    match = re.search(exp_pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0

def calculate_skills_score(text):
    words = set(word_tokenize(text.lower()))
    return len(words & SKILL_KEYWORDS) / len(SKILL_KEYWORDS)

def calculate_project_score(text):
    words = set(word_tokenize(text.lower()))
    return len(words & PROJECT_KEYWORDS) / len(PROJECT_KEYWORDS)

def preprocess_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    experience = extract_experience(text)
    skills_score = calculate_skills_score(cleaned_text)
    project_score = calculate_project_score(cleaned_text)
    return {
        'resume_text': cleaned_text,
        'skills_score': skills_score,
        'experience_years': experience,
        'project_relevance_score': project_score
    }

def create_features(df):
    df['skills_score'] = df['resume_text'].apply(calculate_skills_score)
    df['experience_years'] = df['resume_text'].apply(extract_experience)
    df['project_relevance_score'] = df['resume_text'].apply(calculate_project_score)
    return df

def train_model(X, y):
##   to train and return a Logistic Regression model.
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', TfidfVectorizer(max_features=100), 'resume_text'),
            ('num', StandardScaler(), ['skills_score', 'experience_years', 'project_relevance_score'])
        ]
    )
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
##    to evaluate model and print metrics.
    y_pred = model.predict(X_test)
    """print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))"""

def predict_shortlist(model, pdf_folder):
##  to predict shortlist for all PDFs in folder.
    results = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            features = preprocess_resume(os.path.join(pdf_folder, pdf_file))
            X_new = pd.DataFrame([features])
            prediction = model.predict(X_new)
            results.append((pdf_file, 'Shortlisted' if prediction[0] else 'Rejected'))
    return results

if __name__ == "__main__":
    try:
        train_df = pd.read_csv('training_data.csv')
        train_df = create_features(train_df)
        X = train_df[['resume_text', 'skills_score', 'experience_years', 'project_relevance_score']]
        y = train_df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

        if os.path.exists('resumes'):
         predictions = predict_shortlist(model, 'resumes')
         print("\nShortlist Predictions:")
         for pdf, decision in predictions:
             print(f"{pdf}: {decision}")
        else:
            print("No 'resumes' folder found for prediction.")
    except FileNotFoundError:
        print("training_data.csv not found. Please ensure the file exists.")



















