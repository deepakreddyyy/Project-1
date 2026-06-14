import os
import sys
import joblib

# Import the preprocessing function from preprocess.py
try:
    from preprocess import preprocess_text
except ImportError:
    print("Error: preprocess.py not found in the current directory. Make sure it is present.")
    sys.exit(1)

def main():
    model_path = "best_emotion_model.pkl"
    
    # Check if the trained model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
        
    print(f"Loading trained emotion detection model ('{model_path}')...")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("      AI-Powered Emotion Detection - Text Predictor")
    print("="*60)
    print("This model classifies text into: happy, sad, angry, surprised, or neutral.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # Interactive Loop
    while True:
        try:
            user_input = input("Enter a sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting session. Goodbye!")
            break
            
        # 1. Preprocess the input sentence using our pipeline
        cleaned_text = preprocess_text(user_input, method='stem')
        
        # 2. Predict the label using the loaded model pipeline
        # The pipeline automatically extracts features (TF-IDF) and makes predictions
        try:
            prediction = model.predict([cleaned_text])[0]
            print(f"-> Predicted Emotion: {prediction.upper()}\n")
        except Exception as e:
            print(f"Error during prediction: {e}\n")

if __name__ == "__main__":
    main()
