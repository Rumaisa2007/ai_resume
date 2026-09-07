AI Resume Analyzer

An AI-powered Resume Analyzer built with Python, featuring a Tkinter-based graphical user interface and a Machine Learning/NLP backend.
The application extracts text from PDF resumes, preprocesses the content using NLP techniques, and uses a trained Logistic Regression model with TF-IDF vectorization to analyze and classify resumes.

Features:

* 📄 Upload and read resumes in PDF Format
* 🔍 Extract text automatically from resumes
* 🧹 NLP-based text preprocessing
* 🛑 Stopword removal using NLTK
* 📊 TF-IDF feature extraction
* 🤖 Machine Learning classification using Logistic Regression
* 🖥️ User-friendly GUI built with Tkinter
* 📈 Model evaluation using:
  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix
* ⚡ Fast resume analysis through a desktop application


Project Architecture 

                ┌─────────────────────┐
                │     Tkinter GUI     │
                │    User Interface   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    PDF Resume       │
                │    Text Extraction  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   NLP Preprocessing │
                │ Tokenization        │
                │ Stopword Removal    │
                │ Text Cleaning       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  TF-IDF Vectorizer  │
                │ Feature Extraction  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Logistic Regression │
                │   ML Classification │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Resume Analysis   │
                │      Result         │
                └─────────────────────┘
Technologies used

Programming Language : Python

GUI : Tkinter

Machine Learning:
 Scikit-learn
* Logistic Regression
* TF-IDF Vectorization
* Train/Test Split
* StandardScaler
* Pipeline
* ColumnTransformer
  
Natural Language Processing:
* NLTK
* Tokenization
* Stopword Removal
* Text Preprocessing

Data Processing:
Pandas and NumPy

PDF Processing:
PyPDF2

Libraries used:

The project uses the following Python libraries:
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
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import nltk


##  How It Works

### 1. Resume Upload

The user selects a resume through the **Tkinter GUI**.

The application accepts resumes in PDF format.

### 2. PDF Text Extraction

The application uses `PyPDF2` to read the uploaded PDF and extract its text.

```python
reader = PdfReader(file_path)

text = ""

for page in reader.pages:
    text += page.extract_text()
```

### 3. Text Preprocessing

The extracted resume text is cleaned before being passed to the ML model.

The preprocessing stage can include:

* Converting text to lowercase
* Removing unnecessary characters
* Tokenization
* Removing stopwords
* Cleaning whitespace

NLTK resources are downloaded when required:

```python
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

### 4. Feature Extraction

The cleaned resume text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF helps the model identify words that are important within the resume while reducing the importance of very common words.

```python
vectorizer = TfidfVectorizer()
```

### 5. Machine Learning Model

A **Logistic Regression** classifier is trained using the extracted TF-IDF features.

```python
model = LogisticRegression()
```

The dataset is divided into training and testing sets using:

```python
train_test_split()
```

### 6. Resume Classification

After training, the model receives the processed resume text and predicts its corresponding category/class.

### 7. Model Evaluation

The model performance is evaluated using:

```python
accuracy_score()
precision_score()
recall_score()
confusion_matrix()
```

These metrics help determine how effectively the model classifies resumes.

---

## 📊 Machine Learning Pipeline

The overall ML pipeline is:

Resume Dataset
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train/Test Split
      │
      ▼
Logistic Regression
      │
      ▼
Prediction
      │
      ▼
Model Evaluation


## 🖥️ User Interface

The application uses **Tkinter** to provide a desktop-based graphical interface.

The GUI allows users to:

1. Select a resume
2. Upload the PDF
3. Process the resume
4. Run the trained ML model
5. Display the analysis/prediction

This makes the project accessible without requiring users to interact directly with Python code or the command line.

Suggested Project Structure

AI-Resume-Analyzer/
│
├── dataset/
│   └── training_data.csv
│
├── models/
│   └── resume_model.py
├── app.py
├── README.md

##  Model Evaluation

The project evaluates the trained classifier using multiple performance metrics.

| Metric           | Purpose                                                     |
| ---------------- | ----------------------------------------------------------- |
| Accuracy         | Measures overall correct predictions                        |
| Precision        | Measures how many predicted positives are actually positive |
| Recall           | Measures how many actual positives are correctly identified |
| Confusion Matrix | Shows correct and incorrect predictions for each class      |

Example:
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

cm = confusion_matrix(y_test, y_pred)

Project Objectives

The main objectives of this project are:

* Automate basic resume analysis
* Apply NLP techniques to unstructured resume text
* Convert text into machine-readable features
* Build a supervised Machine Learning classification model
* Provide an easy-to-use desktop interface
* Demonstrate the practical application of ML and NLP in recruitment-related task.


## 👨‍💻 Project Highlights

This project demonstrates practical implementation of:

**Python + Tkinter + NLP + Machine Learning + PDF Processing**

It combines a desktop GUI with a complete text-processing and classification pipeline to create an end-to-end AI resume analysis application.

