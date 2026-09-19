# 🇲🇦 Moroccan Darija Sentiment Analysis

A Natural Language Processing (NLP) project for **sentiment analysis of Moroccan Arabic (Darija) social media texts**.

The project focuses on automatically classifying tweets into three sentiment categories:

* 🟢 **Positive**
* 🔴 **Negative**
* ⚪ **Neutral**

The system combines **ArabicBERT embeddings** with several Machine Learning and Deep Learning models and provides a **Flask web application** for real-time sentiment prediction.

---

## 📌 Project Overview

Social media contains a large amount of user-generated content expressing opinions, emotions, and reactions. However, sentiment analysis becomes particularly challenging for **Moroccan Darija** because of:

* Lack of standardized spelling
* Arabic and Latin script variations
* Frequent use of numbers in Darija writing
* Code-switching between Darija, Arabic, French, Spanish, and Amazigh
* Informal expressions and abbreviations
* Limited labeled datasets for Moroccan Darija

This project aims to develop a sentiment classification system capable of handling these linguistic challenges.

---

## 🎯 Objectives

The main objectives of this project are:

* Collect and prepare Moroccan Darija textual data.
* Clean and normalize social media texts.
* Extract contextual representations using **ArabicBERT**.
* Train and compare different classification models.
* Evaluate models using standard classification metrics.
* Select a suitable model for deployment.
* Develop a web application for real-time sentiment prediction.

---

## 🚧 System Architecture

The project follows the following pipeline:

```text
Social Media Text
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Text Normalization
       │
       ▼
Tokenization
       │
       ▼
ArabicBERT Embeddings
       │
       ├───────────────┬───────────────┬───────────────┐
       ▼               ▼               ▼               ▼
     LSTM            BiLSTM            GRU             CNN
       │               │               │               │
       └───────────────┴───────────────┴───────────────┘
                               │
                               ▼
                              SVM
                               │
                               ▼
                         Model Evaluation
                               │
                               ▼
                       Selected Model
                               │
                               ▼
                       Flask Web Application
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### NLP & Machine Learning

* ArabicBERT / AraBERT
* PyTorch
* Transformers
* Scikit-learn
* NLTK
* PyArabic

### Data Processing

* NumPy
* Pandas
* Regular Expressions

### Data Visualization

* Matplotlib
* Seaborn

### Web Application

* Flask
* Flask-CORS
* HTML / CSS / JavaScript

### Development & Version Control

* Jupyter Notebook / Google Colab
* Git
* GitHub

---


## 📊 Dataset

The project uses the Moroccan Arabic Corpus (MAC) containing Moroccan Darija tweets.

After data cleaning:

Information	Value
Number of entries	17,441
Number of columns	2
Positive	9,894
Negative	3,508
Neutral	4,039

The dataset contains three sentiment classes:

Positive
Negative
Neutral
### Class Distribution (After Cleaning)
Positive : 9,894
Negative : 3,508
Neutral  : 4,039

The original dataset contained 17,444 entries. Three entries containing missing values were removed during cleaning. Duplicate entries were retained because their removal resulted in lower model performance during experimentation

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes several steps adapted to Moroccan Darija:

1. Text cleaning
2. Removal of unwanted characters
3. Normalization of Arabic text
4. Handling of informal writing variations
5. Stop-word processing
6. Tokenization
7. Padding / truncation
8. Generation of contextual embeddings using ArabicBERT

These steps help reduce the linguistic variability commonly found in Moroccan Darija social media content.

---
## 🤖 Tested Models

The following classification models were trained and evaluated using the representations generated from ArabicBERT:

LSTM — Long Short-Term Memory
BiLSTM — Bidirectional Long Short-Term Memory
GRU — Gated Recurrent Unit
CNN — Convolutional Neural Network
SVM — Support Vector Machine

The dataset was divided into training, validation, and test sets. The same data splitting strategy was applied to the different tested models.


## 📈 Results

Five classification models were evaluated using representations generated from ArabicBERT:

| Model  | Accuracy | Precision |  Recall | F1-Score |
| ------ | -------: | --------: | ------: | -------: |
| LSTM   |      83% |       82% |     83% |      83% |
| BiLSTM |  **85%** |   **85%** | **85%** |  **85%** |
| GRU    |      79% |       81% |     79% |      80% |
| CNN    |      84% |       84% |     84% |      84% |
| SVM    |      80% |       80% |     80% |      80% |

The **BiLSTM** model was selected for the final web application based on the comparative evaluation.

---

## 📊 Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Training and validation curves

The experiments achieved accuracy values ranging from **79% to 85%** across the tested models.

The BiLSTM achieved **85% accuracy** and was selected as the final classification model used by the application.

---
## 🌐 Web Application — DarijaSentiment

DarijaSentiment is a web application developed with Flask for real-time sentiment analysis of Moroccan Darija text.

The application integrates the trained BiLSTM model and allows users to submit text and obtain a predicted sentiment.

### Features

The application allows users to:

Enter or paste Moroccan Darija text.
Analyze the submitted text.
Predict the sentiment automatically.
Classify the text as:
🟢 Positive
🔴 Negative
⚪ Neutral

The text input interface supports texts of up to 5,000 characters.

### Interfaces

The application contains several interfaces:

🏠 Home — Introduction to the DarijaSentiment application.
🔍 Text Analysis — Interface for entering and analyzing text.
 ℹ️ About — Information about the project.
📩 Contact — Contact interface.

