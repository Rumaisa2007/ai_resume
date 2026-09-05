import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
from resume_model import train_model, create_features, predict_shortlist

class ResumeScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Screening App")
        self.model = None

        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        tk.Label(self.frame, text="Training Data CSV File:").grid(row=0, column=0, padx=5, pady=5)
        self.csv_entry = tk.Entry(self.frame, width=50)
        self.csv_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.frame, text="Browse", command=self.browse_csv).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(self.frame, text="Resumes Folder:").grid(row=1, column=0, padx=5, pady=5)
        self.folder_entry = tk.Entry(self.frame, width=50)
        self.folder_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.frame, text="Browse", command=self.browse_folder).grid(row=1, column=2, padx=5, pady=5)

        tk.Button(self.frame, text="Train Model", command=self.train_model).grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(self.frame, text="Predict Shortlist", command=self.predict_shortlist).grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.result_text = tk.Text(self.root, width=80, height=20)
        self.result_text.pack(padx=10, pady=10)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.csv_entry.delete(0, tk.END)
        self.csv_entry.insert(0, file_path)

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        self.folder_entry.delete(0, tk.END)
        self.folder_entry.insert(0, folder_path)

    def train_model(self):
        try:
            csv_file = self.csv_entry.get()
            train_df = pd.read_csv(csv_file)
            train_df = create_features(train_df)
            X = train_df[['resume_text', 'skills_score', 'experience_years', 'project_relevance_score']]
            y = train_df['label']
            self.model = train_model(X, y)
            messagebox.showinfo("Model Trained", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_shortlist(self):
        try:
            folder_path = self.folder_entry.get()
            if self.model is None:
                messagebox.showerror("Error", "Please train the model first!")
                return
            predictions = predict_shortlist(self.model, folder_path)
            self.result_text.delete(1.0, tk.END)
            for pdf, decision in predictions:
                self.result_text.insert(tk.END, f"{pdf}: {decision}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeScreeningApp(root)
    root.mainloop()
