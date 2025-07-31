# YouTube Comment Sentiment Analysis: Project Overview

This project analyzes Reddit comments for sentiment classification using natural language processing (NLP) and machine learning techniques. The workflow is organized into two main Jupyter notebooks:

## 1. 1_Preprocessing_&_EDA.ipynb

**Purpose:**
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)

**Key Steps:**
- Loads the Reddit comments dataset from a public URL.
- Cleans the data by removing nulls, duplicates, and empty comments.
- Standardizes text (lowercasing, removing whitespace, newlines, and non-English characters).
- Adds features: word count, character count, stop word count, punctuation count.
- Visualizes data distributions (class balance, word/stopword distributions, boxplots, KDE plots).
- Extracts and visualizes most common words, bigrams, and trigrams.
- Removes stopwords (with some exceptions for sentiment), and applies lemmatization.
- Generates word clouds and stacked bar plots for word frequencies by sentiment category.

**Outcome:**
- Provides a clean, feature-rich dataset ready for modeling.
- Offers insights into comment length, vocabulary, and class imbalance.

## 2. 2_experiment_1_baseline_model.ipynb

**Purpose:**
- Build and evaluate a baseline sentiment classification model.
- Track experiments using MLflow.

**Key Steps:**
- Installs and configures MLflow for experiment tracking (with remote tracking URI).
- Loads and preprocesses the dataset (mirroring steps from the first notebook).
- Vectorizes comments using Bag of Words (CountVectorizer).
- Splits data into training and test sets.
- Trains a Random Forest classifier as a baseline model.
- Logs parameters, metrics, and artifacts (plots, datasets, models) to MLflow.
- Evaluates model performance (accuracy, classification report, confusion matrix).
- Saves the processed dataset for future use.

**Outcome:**
- Establishes a baseline for sentiment classification.
- Provides experiment tracking and reproducibility via MLflow.

---

## How to Use
1. Run `1_Preprocessing_&_EDA.ipynb` to explore and preprocess the data.
2. Run `2_experiment_1_baseline_model.ipynb` to train and evaluate the baseline model, and log results to MLflow.

---

**Note:**
- The project uses Python, pandas, scikit-learn, NLTK, seaborn, matplotlib, and MLflow.
- For MLflow tracking, ensure the tracking server URI is accessible.
- The workflow can be extended with more advanced models and further analysis.
