# AI-Powered Emotion Detection from Text

This project focuses on using Natural Language Processing (NLP) to detect the emotional tone of text. The model classifies sentences or paragraphs into one of five categories: **happy, sad, angry, surprised, and neutral**.

This project contains a complete, beginner-friendly machine learning pipeline, a command-line interface, a Jupyter Notebook, and a beautiful web application interface.

---

## 🚀 Project Features
1.  **Text Preprocessing Pipeline**: Automatically cleans text by converting to lowercase, removing noise (punctuation, URLs, handles), tokenizing, removing stopwords, and stemming words to their root form.
2.  **Model Selection & Training**: Trains and compares **Naive Bayes** and **Logistic Regression** classifiers using both **Bag-of-Words** and **TF-IDF** features.
3.  **Saved Model**: The best combination (**Logistic Regression + TF-IDF**, achieving **53.46% accuracy** on the 40,000 tweet dataset) is saved for immediate use.
4.  **Jupyter Notebook**: Step-by-step interactive demonstration of the code.
5.  **Interactive CLI Tool**: A command-line script to test custom inputs.
6.  **Web Application Interface**: A modern, responsive Flask-based web application with a glassmorphism design, custom animations, and emotion-themed colors/emojis.

---

## 📁 Repository Structure
*   `preprocess.py`: Contains the text cleaning, tokenization, stopword removal, and stemming functions.
*   `download_data.py`: Downloads the raw CrowdFlower Twitter dataset and prepares the clean dataset.
*   `train.py`: Preprocesses the data, trains the models, compares feature extraction methods, and exports the best model.
*   `predict.py`: Interactive command-line prediction interface.
*   `app.py`: Flask web application backend API.
*   `static/index.html`: Web application frontend interface (HTML, CSS, JS).
*   `emotion_detection.ipynb`: Step-by-step Jupyter Notebook.
*   `report.md`: Detailed final project report.
*   `requirements.txt`: Python package dependencies.
*   `best_emotion_model.pkl`: Exported model pipeline.

---

## 🛠️ How to Run the Project

### Step 1: Install Dependencies
First, install all required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Download and Clean the Dataset
Run the data collection script to download the dataset and create the mapped emotions:
```bash
python download_data.py
```

### Step 3: Train and Evaluate Models
To train the classifiers and export the best model, run:
```bash
python train.py
```
*(If Matplotlib cannot import due to antivirus security restrictions, the script will output the confusion matrix in a clean text-based format instead of crashing)*.

### Step 4: Run CLI Predictions
To predict emotions in your terminal:
```bash
python predict.py
```
Type any sentence when prompted (e.g. *"I am so happy and excited!"*) and type `exit` to quit.

### Step 5: Launch the Web App
To start the interactive web page:
```bash
python app.py
```
Then, open your web browser and go to:


Enter your text in the text area and click **Detect Emotion** to watch the real-time classification with custom animations and emojis!
