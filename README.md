# SAMSUNG INNOVATION CAMPUS (SIC - AI)
As a student of the Samsung Innovation Campus (SIC), I have been participating in a global educational initiative by Samsung that aims to empower learners with essential skills in Artificial Intelligence (AI). The AI track I am following provides a comprehensive curriculum that includes both foundational and advanced topics such as mathematics for AI, Python programming, data preprocessing, machine learning, deep learning, and hands-on project development.

This repository contains all coursework exercises as well as the capstone project completed during the program.

---
# Capston Project: Toxic Comment Detection

This project addresses the task of detecting toxic comments in user-generated text using both traditional machine learning algorithms and deep learning approaches. The primary goal is to evaluate the effectiveness of various models in classifying toxic content with high accuracy and robustness.

## Overview

The detection of toxic comments is a crucial task in content moderation for online platforms. This project implements a full pipeline including text preprocessing, feature engineering, model training, and performance evaluation.

Models implemented:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Long Short-Term Memory network (LSTM)

The best performance was achieved using the LSTM model, which reached **97.3% accuracy** on the validation set.

## Dataset

- **Source:** [Kaggle - Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Type:** Multi-label classification (e.g., toxic, obscene, insult, threat)
- **Preprocessing:** Included tokenization, lowercasing, stopword removal, and sequence padding for deep learning models.

## Methodology

1. **Data Preprocessing**
   - Tokenization and normalization
   - Stopword removal
   - Feature extraction: TF-IDF (for classical models), sequence embeddings (for LSTM)

2. **Model Training**
   - Traditional models trained using scikit-learn and XGBoost
   - LSTM model implemented using Keras with TensorFlow backend

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Confusion matrix analysis for detailed error inspection

## Results

| Model                  | Validation Accuracy |
|------------------------|---------------------|
| Multinomial Naive Bayes| 89.2%               |
| Logistic Regression    | 91.8%               |
| SVM                    | 93.0%               |
| XGBoost                | 95.1%               |
| **LSTM**               | **97.3%**           |

