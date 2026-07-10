import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

def main():
    print("Starting Model Training...")
    
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base_dir, 'dataset', 'fake_review_dataset.csv')
    model_save_path = os.path.join(base_dir, 'model', 'model.pkl')
    tfidf_save_path = os.path.join(base_dir, 'model', 'tfidf.pkl')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Rename text column
    if 'text_' in df.columns:
        df = df.rename(columns={'text_': 'review'})
    
    # Encode labels: OR -> 0 (Genuine), CG -> 1 (Fake)
    print("Encoding labels...")
    df['label'] = df['label'].map({'OR': 0, 'CG': 1})
    
    # Drop any rows with missing reviews or labels
    df = df.dropna(subset=['review', 'label'])
    
    # Clean review text
    print("Cleaning text...")
    df['clean_review'] = df['review'].apply(utils.clean_text)
    
    # Features and Labels
    X = df['clean_review']
    y = df['label']
    
    # Train/Test Split
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization
    print("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Logistic Regression Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate accuracy
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Training Complete.")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save Model and Vectorizer
    print("Saving model and vectorizer...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(tfidf_save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Saved model.pkl and tfidf.pkl to model/ directory.")
    print("Stage 1 completed successfully.")

if __name__ == "__main__":
    main()
