import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Try importing matplotlib and seaborn, but fail gracefully if blocked by antivirus/Defender
matplotlib_available = True
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"\nWarning: Could not import matplotlib/seaborn due to system/antivirus restriction: {e}")
    print("We will skip generating the visual plot but will display the confusion matrix in text format.\n")
    matplotlib_available = False

# Import preprocessing function from our preprocess.py
from preprocess import preprocess_text

def train_and_evaluate():
    clean_data_path = "emotions_clean.csv"
    if not os.path.exists(clean_data_path):
        print(f"Error: Cleaned dataset not found at {clean_data_path}. Please run download_data.py first.")
        return

    print("Step 1: Loading cleaned dataset...")
    df = pd.read_csv(clean_data_path)
    
    # Handle missing values if any
    df = df.dropna(subset=['text', 'label'])
    
    print(f"Dataset contains {len(df)} samples.")
    print("Class distribution:\n", df['label'].value_counts())

    print("\nStep 2: Preprocessing all text samples (applying tokenization, cleaning, stopword removal, and stemming)...")
    # For speed and progress feedback, we can print progress updates
    total_samples = len(df)
    # Using simple chunking or batching to print progress
    df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, method='stem'))
    print("Preprocessing completed!")

    # Split dataset into training and testing sets (80% train, 20% test)
    print("\nStep 3: Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # We will try both Bag-of-Words (CountVectorizer) and TF-IDF (TfidfVectorizer)
    print("\nStep 4: Evaluating Feature Extraction and Models...")
    
    vectorizers = {
        'Bag-of-Words': CountVectorizer(),
        'TF-IDF': TfidfVectorizer()
    }
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    best_accuracy = 0.0
    best_pipeline = None
    best_config_name = ""

    # Test combinations
    for vec_name, vec in vectorizers.items():
        for model_name, model in models.items():
            print(f"\n--- Training {model_name} with {vec_name} ---")
            
            # Create a Scikit-Learn pipeline to bundle vectorizer and model together
            pipeline = Pipeline([
                ('vectorizer', vec),
                ('classifier', model)
            ])
            
            # Train the pipeline
            pipeline.fit(X_train, y_train)
            
            # Make predictions on test set
            predictions = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (Weighted): {f1:.4f}")
            print(f"Classification Report:\n", classification_report(y_test, predictions))
            
            # Track the best performing pipeline
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pipeline = pipeline
                best_config_name = f"{model_name} + {vec_name}"

    print(f"\n==================================================")
    print(f"Best Configuration: {best_config_name} with Accuracy: {best_accuracy:.4f}")
    print(f"==================================================")

    # Re-evaluate the best model to generate confusion matrix
    print("\nStep 5: Generating Confusion Matrix for the best model...")
    best_predictions = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_predictions)
    labels = sorted(df['label'].unique())

    print("\nText-based Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels], columns=[f"Predicted: {l}" for l in labels])
    print(cm_df.to_string())

    if matplotlib_available:
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix - {best_config_name}')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plot_file = 'confusion_matrix.png'
            plt.savefig(plot_file)
            print(f"Saved confusion matrix plot to {plot_file}")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to plot confusion matrix: {e}")
    else:
        print("\nNote: Visual confusion matrix image was not generated because matplotlib/seaborn is not available on this system due to security/antivirus settings.")

    # Save the best pipeline using joblib so we can load it easily in predict.py
    model_file = 'best_emotion_model.pkl'
    print(f"\nStep 6: Exporting the best model pipeline to {model_file}...")
    joblib.dump(best_pipeline, model_file)
    print("Model pipeline saved successfully!")

if __name__ == "__main__":
    train_and_evaluate()
