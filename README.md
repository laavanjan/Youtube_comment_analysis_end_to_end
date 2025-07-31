# YouTube Comment Sentiment Analysis: Project Overview


This project analyzes Reddit comments for sentiment classification using natural language processing (NLP) and machine learning techniques. The workflow is organized into three main Jupyter notebooks:

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


## 3. 3_experiment_2_bow_vs_tfidf.ipynb

**Purpose:**
- Compare the performance of Bag of Words (BoW) and TF-IDF vectorization techniques for sentiment classification.
- Evaluate the impact of different n-gram ranges (unigrams, bigrams, trigrams) on model performance.
- Track all experiments using MLflow.

**Key Steps:**
- Loads the preprocessed dataset.
- Configures MLflow for experiment tracking.
- Defines a reusable experiment function to:
  - Vectorize text using either BoW or TF-IDF with specified n-gram ranges and feature limits.
  - Split data into training and test sets.
  - Train a Random Forest classifier.
  - Log parameters, metrics (accuracy, classification report), confusion matrix plots, and models to MLflow.
- Runs experiments for both BoW and TF-IDF with n-gram ranges of (1,1), (1,2), and (1,3).
- Compares results to determine which vectorization method and n-gram configuration performs best.

**Outcome:**
- Provides a systematic comparison of feature engineering strategies for text classification.
- Helps identify the optimal vectorization approach for sentiment analysis on this dataset.

---


## 4. 4_experiment_3_tfidf_(1,3)_max_features.ipynb

**Purpose:**
- Investigate the effect of varying the `max_features` parameter in TF-IDF vectorization (with trigrams) on sentiment classification performance.
- Identify the optimal number of features for the best model accuracy.
- Track all experiments using MLflow.

**Key Steps:**
- Loads the preprocessed dataset.
- Configures MLflow for experiment tracking.
- Defines an experiment function to:
  - Vectorize text using TF-IDF with ngram_range=(1,3) and different max_features values.
  - Split data into training and test sets.
  - Train a Random Forest classifier.
  - Log parameters, metrics (accuracy, classification report), confusion matrix plots, and models to MLflow.
- Iterates over a range of max_features values (e.g., 1000 to 10000) to compare results.
- Analyzes how feature dimensionality impacts model performance.

**Outcome:**
- Provides insight into the trade-off between feature size and classification accuracy.
- Helps select the best max_features value for TF-IDF-based sentiment models.

---

## How to Use
1. Run `1_Preprocessing_&_EDA.ipynb` to explore and preprocess the data.
2. Run `2_experiment_1_baseline_model.ipynb` to train and evaluate the baseline model, and log results to MLflow.
3. Run `3_experiment_2_bow_vs_tfidf.ipynb` to compare BoW and TF-IDF vectorization methods and analyze their impact on model performance.
4. Run `4_experiment_3_tfidf_(1,3)_max_features.ipynb` to experiment with different max_features settings for TF-IDF trigrams and optimize feature selection.

---

**Note:**
- The project uses Python, pandas, scikit-learn, NLTK, seaborn, matplotlib, and MLflow.
- For MLflow tracking, ensure the tracking server URI is accessible.
- The workflow can be extended with more advanced models and further analysis.
