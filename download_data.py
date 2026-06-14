import pandas as pd
import urllib.request
import os

def download_and_clean_data():
    url = "https://raw.githubusercontent.com/tlkh/text-emotion-classification/master/dataset/original/text_emotion.csv"
    output_raw = "text_emotion_raw.csv"
    output_clean = "emotions_clean.csv"

    print("Step 1: Downloading raw dataset...")
    if not os.path.exists(output_raw):
        try:
            urllib.request.urlretrieve(url, output_raw)
            print(f"Successfully downloaded raw dataset to {output_raw}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        print(f"Raw dataset file '{output_raw}' already exists. Skipping download.")

    print("\nStep 2: Loading raw dataset into Pandas...")
    df = pd.read_csv(output_raw)
    
    print("\nColumns in raw dataset:")
    print(df.columns.tolist())
    
    print("\nRaw emotion (sentiment) distribution:")
    print(df['sentiment'].value_counts())

    # Map raw emotions to simplified categories: happy, sad, angry, surprised, neutral
    emotion_mapping = {
        'happiness': 'happy',
        'fun': 'happy',
        'enthusiasm': 'happy',
        'relief': 'happy',
        'love': 'happy',
        'sadness': 'sad',
        'worry': 'sad',
        'boredom': 'sad',
        'empty': 'sad',
        'anger': 'angry',
        'hate': 'angry',
        'surprise': 'surprised',
        'neutral': 'neutral'
    }

    print("\nStep 3: Pre-processing dataset fields...")
    # Keep only columns we need and rename them
    df_cleaned = df[['sentiment', 'content']].copy()
    df_cleaned.columns = ['raw_sentiment', 'text']
    
    # Map raw sentiment to simplified classes
    df_cleaned['label'] = df_cleaned['raw_sentiment'].map(emotion_mapping)
    
    # Check if there are any unmapped categories
    unmapped = df_cleaned[df_cleaned['label'].isnull()]
    if not unmapped.empty:
        print(f"Warning: Found unmapped raw sentiments: {unmapped['raw_sentiment'].unique()}")
        df_cleaned = df_cleaned.dropna(subset=['label'])
    
    # Keep only the cleaned text and mapped label columns
    df_final = df_cleaned[['text', 'label']].copy()
    
    print("\nCleaned emotion (label) distribution:")
    print(df_final['label'].value_counts())

    print(f"\nStep 4: Saving cleaned dataset to {output_clean}...")
    df_final.to_csv(output_clean, index=False)
    print("Cleaned dataset saved successfully!")

if __name__ == "__main__":
    download_and_clean_data()
