import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (Ensure 'movies.csv' is in the same folder)
try:
    df = pd.read_csv('movies.csv')
except FileNotFoundError:
    messagebox.showerror("Error", "Dataset not found! Make sure 'movies.csv' is in the same folder.")
    exit()

# Check if dataset has required columns
if 'title' not in df.columns or 'genre' not in df.columns:
    messagebox.showerror("Error", "Dataset must have 'title' and 'genre' columns.")
    exit()

# Remove empty values and ensure genres are strings
df = df.dropna(subset=['genre'])
df['genre'] = df['genre'].astype(str).str.lower().str.strip()

# Remove stop words from genres
df['genre'] = df['genre'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))

# Vectorizing the genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movies(title):
    if title not in df['title'].values:
        return []
    
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices].tolist()

# GUI Setup
def on_recommend():
    title = movie_entry.get()
    recommendations = recommend_movies(title)
    if recommendations:
        result_label.config(text="\n".join(recommendations))
    else:
        messagebox.showerror("Error", "Movie not found in the database.")

# Create Tkinter window
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("400x400")

# Widgets
tk.Label(root, text="Enter Movie Title:").pack(pady=10)
movie_entry = ttk.Entry(root, width=40)
movie_entry.pack(pady=5)

recommend_btn = ttk.Button(root, text="Recommend", command=on_recommend)
recommend_btn.pack(pady=10)

result_label = tk.Label(root, text="", justify=tk.LEFT)
result_label.pack(pady=10)

# Run the application
root.mainloop()
