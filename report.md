# Project Report: AI-Powered Emotion Detection from Text

## 1. Project Overview
This project focuses on building an AI model using Natural Language Processing (NLP) to classify text sentences or paragraphs into one of five emotional categories: **happy, sad, angry, surprised, and neutral**.

### Problem Statement
Build an AI model that reads text input from a user and predicts the underlying emotion expressed.

---

## 2. Text Preprocessing Pipeline
Text data in its raw form is unstructured and contains noise (HTML links, special characters, uppercase and lowercase duplication). We built a text preprocessing pipeline in `preprocess.py` to clean the text using the following steps:

1.  **Lowercasing**: Converts all characters to lowercase to prevent duplicates (e.g., "Sadness" vs "sadness").
2.  **Removing Special Characters**: Removes web URLs, Twitter user handles, punctuation, numbers, and excess spacing.
3.  **Tokenization**: Splits the raw text sentences into individual lists of words (tokens) using NLTK.
4.  **Stopwords Removal**: Removes common structural words (e.g., "is", "the", "and") that do not convey emotion.
5.  **Stemming**: Reduces inflected or derived words to their base form (e.g., "feeling", "feels", "felt" -> "feel") using NLTK's `PorterStemmer`.

---

## 3. Feature Extraction
To train machine learning algorithms, text tokens must be transformed into numerical features. We tested and compared two techniques:
*   **Bag-of-Words (BoW)**: Represents text as word count vectors, tracking the frequency of each word in the vocabulary.
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weights words by how frequent they are in the sentence (TF) and scales them down if they appear too frequently across the entire dataset (IDF). This highlights unique and informative emotional terms.

---

## 4. Models Trained & Evaluated
We trained and compared two classification algorithms on both BoW and TF-IDF features using an 80/20 train/test split:
1.  **Naive Bayes Classifier** (using `MultinomialNB`)
2.  **Logistic Regression Classifier** (using `LogisticRegression`)

### Evaluation Results (Accuracy & Weighted F1-Score)
*   **Naive Bayes + Bag-of-Words**: Accuracy = 51.71%, F1-Score = 0.4525
*   **Logistic Regression + Bag-of-Words**: Accuracy = 51.55%, F1-Score = 0.4985
*   **Naive Bayes + TF-IDF**: Accuracy = 51.18%, F1-Score = 0.4259
*   **Logistic Regression + TF-IDF**: Accuracy = 53.46%, F1-Score = 0.5037

### Selected Model
The **Logistic Regression + TF-IDF** combination achieved the highest accuracy (**53.46%**) and was saved using `joblib` as `best_emotion_model.pkl` for interactive predictions.

---

## 5. Text-Based Confusion Matrix (Best Model)
The confusion matrix shows where our best model predicted correctly and where it made mistakes on the test set (8,000 samples):

```text
                   Predicted: angry  Predicted: happy  Predicted: neutral  Predicted: sad  Predicted: surprised
Actual: angry                    23                37                  24             203                     0
Actual: happy                     3              1730                 319             568                     2
Actual: neutral                   3               538                 492             694                     1
Actual: sad                      15               538                 346            2024                     3
Actual: surprised                 2               167                  69             191                     8
```

*Key Observations:*
*   The model is highly accurate at predicting **happy** (1,730 correct) and **sad** (2,024 correct) classes due to having a larger number of training samples for these emotions.
*   The model tends to confuse minor categories like **angry** and **surprised** with **sad** or **happy**, which is expected given the smaller representation of these classes in the dataset.

---

## 6. How to Run the Project

Follow these steps to run the project locally on your machine:

### Step 1: Install Dependencies
Install all required libraries by running:
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset
Download and preprocess the CrowdFlower raw dataset by running:
```bash
python download_data.py
```
This downloads the raw CSV file and generates a cleaned, mapped version named `emotions_clean.csv`.

### Step 3: Train the Models
Train and evaluate the models by running:
```bash
python train.py
```
This scripts trains Naive Bayes and Logistic Regression classifiers, outputs performance reports, and exports the best model pipeline as `best_emotion_model.pkl`.

### Step 4: Run Interactive Predictions via Command Line
Test the model on custom inputs by running the interactive command-line predictor:
```bash
python predict.py
```
Type any sentence when prompted (e.g., *"I am having an amazing day!"*), and the model will output the predicted emotion. Type `exit` to quit.

### Step 5: Launch the Web Application
To start the modern, interactive web interface, run the Flask server:
```bash
python app.py
```
1.  Once started, open your web browser and navigate to: `http://127.0.0.1:5000`
2.  Type your text inside the text area of the web interface.
3.  Click **Detect Emotion** to see the predicted emotion along with custom colors and emojis in real-time.

