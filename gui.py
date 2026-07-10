import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import os
import sys
import sqlite3
from datetime import datetime

# Import custom utils for text cleaning
import utils

class FakeReviewDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake Product Review Detection")
        self.root.geometry("600x650")
        self.root.configure(bg="white")
        
        # Database setup
        self.setup_database()
        
        # Load Model and Vectorizer
        self.model = None
        self.vectorizer = None
        self.load_models()
        
        # Build UI
        self.build_ui()
        
    def setup_database(self):
        base_dir = os.path.dirname(__file__)
        self.db_path = os.path.join(base_dir, 'prediction_history.db')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review TEXT,
                predicted_result TEXT,
                confidence TEXT,
                date_time TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def load_models(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'model', 'model.pkl')
        tfidf_path = os.path.join(base_dir, 'model', 'tfidf.pkl')
        
        try:
            if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
                raise FileNotFoundError("Model files (model.pkl, tfidf.pkl) not found. Please run the training script first.")
                
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            with open(tfidf_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
        except Exception as e:
            messagebox.showerror("Error Loading Model", str(e))
            self.root.destroy()
            sys.exit(1)
            
    def build_ui(self):
        # Title Label
        title_label = tk.Label(self.root, text="Fake Product Review Detection", font=("Arial", 16, "bold"), bg="white")
        title_label.pack(pady=20)
        
        # Input Label
        input_label = tk.Label(self.root, text="Enter Product Review:", font=("Arial", 12), bg="white")
        input_label.pack(anchor="w", padx=30)
        
        # Text Box
        self.review_text = tk.Text(self.root, height=8, width=65, font=("Arial", 11))
        self.review_text.pack(pady=10, padx=30)
        
        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg="white")
        btn_frame.pack(pady=10)
        
        # Buttons
        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict_review, font=("Arial", 12), bg="#0078D7", fg="white", width=10)
        predict_btn.grid(row=0, column=0, padx=10)
        
        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_text, font=("Arial", 12), bg="#0078D7", fg="white", width=10)
        clear_btn.grid(row=0, column=1, padx=10)
        
        history_btn = tk.Button(btn_frame, text="History", command=self.show_history, font=("Arial", 12), bg="#0078D7", fg="white", width=10)
        history_btn.grid(row=0, column=2, padx=10)
        
        exit_btn = tk.Button(btn_frame, text="Exit", command=self.root.destroy, font=("Arial", 12), bg="#0078D7", fg="white", width=10)
        exit_btn.grid(row=0, column=3, padx=10)
        
        # Results Frame
        self.result_frame = tk.Frame(self.root, bg="white")
        self.result_frame.pack(pady=20)
        
        self.prediction_label = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), bg="white")
        self.prediction_label.pack()
        
        self.confidence_label = tk.Label(self.result_frame, text="", font=("Arial", 12), bg="white")
        self.confidence_label.pack()
        
        self.accuracy_label = tk.Label(self.result_frame, text="Model Accuracy: 89.95%", font=("Arial", 10), bg="white", fg="gray")
        self.accuracy_label.pack(pady=10)
        
    def clear_text(self):
        self.review_text.delete(1.0, tk.END)
        self.prediction_label.config(text="")
        self.confidence_label.config(text="")
        
    def save_prediction(self, review, result, confidence):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO prediction_history (review, predicted_result, confidence, date_time)
            VALUES (?, ?, ?, ?)
        ''', (review[:100] + '...' if len(review) > 100 else review, result, confidence, date_time))
        conn.commit()
        conn.close()

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Prediction History")
        history_window.geometry("600x400")
        
        columns = ("ID", "Review", "Result", "Confidence", "Date")
        tree = ttk.Treeview(history_window, columns=columns, show="headings")
        
        tree.heading("ID", text="ID")
        tree.heading("Review", text="Review")
        tree.heading("Result", text="Result")
        tree.heading("Confidence", text="Confidence")
        tree.heading("Date", text="Date")
        
        tree.column("ID", width=30)
        tree.column("Review", width=250)
        tree.column("Result", width=80)
        tree.column("Confidence", width=80)
        tree.column("Date", width=120)
        
        tree.pack(fill="both", expand=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM prediction_history ORDER BY id DESC')
        rows = cursor.fetchall()
        for row in rows:
            tree.insert("", tk.END, values=row)
        conn.close()

    def predict_review(self):
        review = self.review_text.get(1.0, tk.END).strip()
        
        if not review:
            messagebox.showwarning("Input Error", "Please enter a review to predict.")
            return
            
        try:
            # Clean review
            cleaned_review = utils.clean_text(review)
            
            # Transform
            features = self.vectorizer.transform([cleaned_review])
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Results
            if prediction == 1:
                result_text = "Fake Review"
                color = "red"
                confidence = probabilities[1]
            else:
                result_text = "Genuine Review"
                color = "green"
                confidence = probabilities[0]
                
            conf_str = f"{confidence*100:.1f}%"
            
            # Update UI
            self.prediction_label.config(text=f"Prediction: {result_text}", fg=color)
            self.confidence_label.config(text=f"Confidence: {conf_str}")
            
            # Save to Database
            self.save_prediction(review, result_text, conf_str)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", "An error occurred during prediction.")
            print(f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeReviewDetectorGUI(root)
    root.mainloop()
