

# YouTube Comment Sentiment Analysis: Project Overview

This project analyzes YouTube (and Reddit) comments for sentiment classification using natural language processing (NLP) and machine learning techniques. The workflow includes Jupyter notebooks for data science, a Flask API for model serving, and a Chrome extension frontend.

---

## Demo

Below are screenshots of the Chrome extension and API in action:

<div align="center">
<img src="img (5).png" alt="Main Demo" width="500" style="margin:12px 0;">
  <br>
  <img src="img (1).png" alt="Demo 1" width="220" style="margin:8px;">
  <img src="img (2).png" alt="Demo 2" width="220" style="margin:8px;">
  <img src="img (3).png" alt="Demo 3" width="220" style="margin:8px;">
  <img src="img (4).png" alt="Demo 4" width="220" style="margin:8px;">
  <br>
  <img src="img (6).png" alt="Demo 6" width="220" style="margin:8px;">
</div>

### Video Demo

<div align="center">
  <video src="screenRec.mp4" controls width="480" style="margin:16px 0;">
    Your browser does not support the video tag.
  </video>
</div>

---

## API & Frontend Components

### Flask API (`flask_api/main.py`)
- Provides REST endpoints for sentiment prediction.
- Loads the trained model from MLflow Model Registry and a local TF-IDF vectorizer.
- Endpoints:
  - `/predict` (POST): Predicts sentiment for a list of comments.
  - `/predict_with_timestamps` (POST): Predicts sentiment for comments with timestamps.
  - `/generate_chart` (POST): Generates charts (details in code).
- Example request to `/predict`:
  ```json
  {
    "comments": ["This video is awesome"]
  }
  ```
- Returns: List of comments with predicted sentiment.

### Chrome Extension Frontend (`yt-chrome-plugin-frontend/`)
- Contains `manifest.json`, `popup.html`, and `popup.js` for a Chrome extension UI.
- Allows users to interact with the Flask API directly from YouTube.
- UI is now a modern, light theme with color-coded sentiment cards (green for Positive, yellow for Neutral, red for Negative).
- Comments are displayed in clean, minimalist cards with timestamps and sentiment labels.
- Metrics and charts are shown in a visually appealing layout.
- Permissions in `manifest.json` include both `http://localhost/*` and `http://127.0.0.1/*` for local API access.
- See the folder for implementation details and customization options.
---
---

## Data & Model Artifacts
- All processed data is in the `data/` folder (raw and interim CSVs).
- Model artifacts: `lgbm_model.pkl`, `tfidf_vectorizer.pkl`.
- MLflow experiment tracking: `experiment_info.json`, `dvc.yaml`, `dvc.lock`.
---

## Project Structure

- `src/`: Data ingestion, preprocessing, model building, evaluation, and registration scripts.
- `notebooks/`: Jupyter notebooks for EDA, experiments, and model development.
- `flask_api/`: Flask REST API for serving predictions.
- `yt-chrome-plugin-frontend/`: Chrome extension for user interaction.
- `data/`: Raw and processed datasets.
- `requirements.txt`, `setup.py`: Project dependencies and setup.

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


## 5. 5_experiment_4_handling_imbalanced_data.ipynb

**Purpose:**
- Address class imbalance in the sentiment dataset using various resampling and class weighting techniques.
- Evaluate the impact of different imbalance handling methods on model performance.
- Track all experiments using MLflow.

**Key Steps:**
- Loads the preprocessed dataset.
- Configures MLflow for experiment tracking.
- Defines an experiment function to:
  - Vectorize text using TF-IDF with ngram_range=(1,3) and a fixed max_features value.
  - Split data into training and test sets.
  - Apply different imbalance handling strategies to the training set:
    - Class weighting in Random Forest
    - SMOTE (oversampling)
    - ADASYN (oversampling)
    - Random undersampling
    - SMOTEENN (combined over- and under-sampling)
  - Train a Random Forest classifier.
  - Log parameters, metrics (accuracy, classification report), confusion matrix plots, and models to MLflow.
- Compares the results of each imbalance handling method.

**Outcome:**
- Provides insight into the effectiveness of different strategies for handling imbalanced data in sentiment classification.
- Helps select the best approach for improving model performance on minority classes.

---


## 6. 6_experiment_5_xgboost_with_hpt.ipynb

**Purpose:**
- Compare multiple machine learning algorithms for sentiment classification, including XGBoost, Logistic Regression, SVM, LightGBM, KNN, Naive Bayes, and Random Forest.
- Use Optuna for hyperparameter tuning of XGBoost to optimize model performance.
- Address class imbalance using SMOTE.
- Track all experiments using MLflow.

**Key Steps:**
- Loads the preprocessed dataset.
- Remaps class labels for compatibility with XGBoost.
- Splits data into training and test sets.
- Vectorizes text using TF-IDF with trigrams and a fixed max_features value.
- Applies SMOTE to balance the training data.
- Defines a logging function to record model parameters, metrics, and artifacts in MLflow.
- Compares several algorithms by training and evaluating each, logging results to MLflow.
- Uses Optuna to tune XGBoost hyperparameters (n_estimators, learning_rate, max_depth), selects the best model, and logs it.

**Outcome:**
- Provides a comprehensive comparison of popular ML algorithms for sentiment analysis.
- Identifies the best-performing model and hyperparameters for the task.
- Demonstrates the use of automated hyperparameter optimization and experiment tracking.

---


## 7. 7_experiment_6_lightgbm_detailed_hpt.ipynb

**Purpose:**
- Perform detailed hyperparameter tuning for LightGBM using Optuna to maximize sentiment classification performance.
- Address class imbalance using SMOTE.
- Track all experiments and hyperparameter trials using MLflow.

**Key Steps:**
- Loads the preprocessed dataset and remaps class labels for compatibility.
- Vectorizes text using TF-IDF with trigrams and a fixed max_features value.
- Applies SMOTE to balance the dataset.
- Splits the data into training and test sets after resampling.
- Defines a logging function to record model parameters, metrics, and artifacts in MLflow for each Optuna trial.
- Uses Optuna to search a wide hyperparameter space for LightGBM (n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, colsample_bytree, subsample, reg_alpha, reg_lambda).
- Logs each trial and the best model to MLflow, and visualizes parameter importance and optimization history.

**Outcome:**
- Identifies the best hyperparameters for LightGBM on this sentiment analysis task.
- Demonstrates advanced experiment tracking and hyperparameter optimization workflows.

---


## 8. 8_stacking.ipynb

**Purpose:**
- Implement a stacking ensemble model for sentiment classification using LightGBM and Logistic Regression as base learners, and KNN as the meta-learner.
- Compare the performance of the stacking model with individual models.

**Key Steps:**
- Loads the preprocessed dataset and drops rows with missing values.
- Vectorizes text using TF-IDF with trigrams and a fixed max_features value.
- Splits the data into training and test sets.
- Defines LightGBM and Logistic Regression as base models, and KNN as the meta-learner.
- Constructs a StackingClassifier with 5-fold cross-validation.
- Trains the stacking model and evaluates its performance on the test set.
- Compares the stacking model's results to those of the LightGBM model alone.

**Outcome:**
- Demonstrates the use of ensemble learning to potentially improve sentiment classification performance.
- Provides insight into the complexity and benefits of stacking versus single-model approaches.

---

## How to Use
1. Run `1_Preprocessing_&_EDA.ipynb` to explore and preprocess the data.
2. Run `2_experiment_1_baseline_model.ipynb` to train and evaluate the baseline model, and log results to MLflow.
3. Run `3_experiment_2_bow_vs_tfidf.ipynb` to compare BoW and TF-IDF vectorization methods and analyze their impact on model performance.
4. Run `4_experiment_3_tfidf_(1,3)_max_features.ipynb` to experiment with different max_features settings for TF-IDF trigrams and optimize feature selection.
5. Run `5_experiment_4_handling_imbalanced_data.ipynb` to explore and compare different techniques for handling class imbalance in the dataset.
6. Run `6_experiment_5_xgboost_with_hpt.ipynb` to compare ML algorithms and tune XGBoost with Optuna for best performance.
7. Run `7_experiment_6_lightgbm_detailed_hpt.ipynb` to perform detailed hyperparameter tuning for LightGBM with Optuna and MLflow.
8. Run `8_stacking.ipynb` to experiment with stacking ensemble models for sentiment classification.

---

**Note:**
- The project uses Python, pandas, scikit-learn, NLTK, seaborn, matplotlib, MLflow, Flask, and Chrome extension technologies.
- For MLflow tracking, ensure the tracking server URI is accessible.
- The workflow can be extended with more advanced models, APIs, and frontend features.
