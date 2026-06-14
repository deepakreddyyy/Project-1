import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load stopwords list
stop_words = set(stopwords.words('english'))

def preprocess_text(text, method='lemmatize'):
    """
    Cleans and preprocesses a given text string.
    
    Steps:
    1. Lowercasing: Convert text to lowercase.
    2. Removing special characters: Strip out URLs, twitter handles (@user), punctuation, and numbers.
    3. Tokenization: Split text into individual words.
    4. Stopwords removal: Filter out common words that don't carry emotional meaning.
    5. Stemming or Lemmatization: Reduce words to their root form.
    """
    # 1. Lowercasing
    text = str(text).lower()
    
    # 2. Removing special characters (URLs, handles, numbers, punctuation)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove twitter handles (e.g., @tiffanylue)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters, numbers, and punctuation, keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopwords removal
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming or Lemmatization
    if method == 'stem':
        # E.g., "running", "runs" -> "run"
        final_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif method == 'lemmatize':
        # E.g., "better" -> "good", "running" -> "running" (contextual)
        final_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        final_tokens = filtered_tokens
        
    # Reconstruct the tokens back into a single cleaned sentence
    cleaned_text = ' '.join(final_tokens)
    return cleaned_text

# Simple test to verify the preprocessing script works
if __name__ == "__main__":
    sample_text = "I am feeling SO excited about this! Check out https://google.com @user123. It's awesome!"
    print("Original text:", sample_text)
    
    print("\nPreprocessed (with Stemming):")
    print(preprocess_text(sample_text, method='stem'))
    
    print("\nPreprocessed (with Lemmatization):")
    print(preprocess_text(sample_text, method='lemmatize'))
