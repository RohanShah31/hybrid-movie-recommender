import json
import os

def create_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
        if source and source[-1] == '\n':
            source[-1] = source[-1][:-1]
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = [
    ("markdown", "# Hybrid Movie Recommendation System with Cold-Start Handling\n\nThis notebook demonstrates a hybrid recommendation engine combining Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF on genres), dynamically weighted based on user interaction count to elegantly solve the Cold-Start problem.\n\n*Note: This notebook is fully standalone and can be run in Google Colab.*"),
    
    ("markdown", "## 1. Setup and Install Dependencies\nWe'll make sure the required libraries are installed (especially useful if running in Colab)."),
    
    ("code", """!pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn requests"""),
    
    ("markdown", "## 2. Download Dataset\nThis cell downloads the MovieLens 1M dataset directly into the notebook environment."),
    
    ("code", """import os
import urllib.request
import zipfile
import ssl

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
zip_path = os.path.join(data_dir, "ml-1m.zip")

ssl._create_default_https_context = ssl._create_unverified_context

if not os.path.exists(zip_path):
    print("Downloading MovieLens 1M dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
print("Dataset ready in data/ml-1m/")
"""),
    
    ("markdown", "## 3. Data Preprocessing\nHere we define functions to load and clean the dataset, as well as calculate user and movie interaction counts."),
    
    ("code", """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

def load_and_preprocess_data():
    movies_path = os.path.join(data_dir, "ml-1m", "movies.dat")
    ratings_path = os.path.join(data_dir, "ml-1m", "ratings.dat")
    
    movies = pd.read_csv(movies_path, sep='::', engine='python', encoding='latin-1',
                         names=['movie_id', 'title', 'genres'])
    ratings = pd.read_csv(ratings_path, sep='::', engine='python', encoding='latin-1',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Calculate interactions
    user_counts = ratings.groupby('user_id').size().reset_index(name='user_interaction_count')
    ratings = pd.merge(ratings, user_counts, on='user_id')
    
    # Popularity baseline
    popularity = ratings.groupby('movie_id').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    movies = pd.merge(movies, popularity, on='movie_id', how='left')
    movies['num_ratings'] = movies['num_ratings'].fillna(0)
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    
    return movies, ratings

movies, ratings = load_and_preprocess_data()
print(f"Total Users: {ratings['user_id'].nunique()}")
print(f"Total Movies: {movies['movie_id'].nunique()}")
"""),

    ("markdown", "## 4. Train/Test Split\nWe will use 80% of ratings for training our models, and 20% to evaluate our recommendations."),
    
    ("code", """train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
print(f"Training ratings: {len(train_ratings)}")
print(f"Testing ratings: {len(test_ratings)}")

train_user_counts = train_ratings['user_id'].value_counts()
cold_users = train_user_counts[train_user_counts < 5].index.tolist()
warm_users = train_user_counts[train_user_counts >= 5].index.tolist()

print(f"Number of Cold-Start Users (< 5 interactions in train): {len(cold_users)}")
print(f"Number of Warm Users (>= 5 interactions in train): {len(warm_users)}")
"""),

    ("markdown", "## 5. Collaborative Filtering Model (SVD)\nBuilds the user-item matrix using `scikit-surprise`."),
    
    ("code", """from surprise import SVD, Dataset, Reader

class CollaborativeFilteringSVD:
    def __init__(self, n_factors=50):
        self.model = SVD(n_factors=n_factors, random_state=42)

    def fit(self, train_df):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)
        self.model.fit(data.build_full_trainset())

    def predict_score(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est

cf_model = CollaborativeFilteringSVD(n_factors=50)
print("Training SVD Model...")
cf_model.fit(train_ratings)
print("SVD Training Complete.")
"""),

    ("markdown", "## 6. Content-Based Filtering Model\nUses TF-IDF on movie genres and calculates Cosine Similarity."),
    
    ("code", """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
        
    def fit(self, movies_df):
        self.movies_df = movies_df.copy()
        self.movie_indices = pd.Series(self.movies_df.index, index=self.movies_df['movie_id']).to_dict()
        tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['genres'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def predict_score(self, user_liked_movies, target_movie_id):
        if target_movie_id not in self.movie_indices: return 0.0
        target_idx = self.movie_indices[target_movie_id]
        
        similarities = []
        for liked_id in user_liked_movies:
            if liked_id in self.movie_indices:
                liked_idx = self.movie_indices[liked_id]
                similarities.append(self.similarity_matrix[target_idx][liked_idx])
                
        if not similarities: return 0.0
        # Scale 0-1 similarity to 1-5 score
        return 1 + (np.mean(similarities) * 4)

cb_model = ContentBasedFiltering()
print("Building Content-Based Model...")
cb_model.fit(movies)
print("Content-Based Model Ready.")
"""),

    ("markdown", "## 7. Hybrid Recommender\nDynamically blends CF and Content scores based on user interaction count."),
    
    ("code", """class HybridRecommender:
    def __init__(self, cf_model, cb_model, movies_df, c_weight=5):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.movies_df = movies_df
        self.c_weight = c_weight
        
    def recommend(self, user_id, train_ratings_df, n=10):
        user_history = train_ratings_df[train_ratings_df['user_id'] == user_id]
        interaction_count = len(user_history)
        
        # New User Fallback
        if interaction_count == 0:
            popular = self.movies_df.sort_values(by=['num_ratings', 'avg_rating'], ascending=[False, False]).head(n)
            popular['cf_weight'] = 0
            popular['final_score'] = popular['avg_rating']
            popular['method'] = 'popularity'
            return popular
            
        liked_movies = user_history[user_history['rating'] >= 4]['movie_id'].tolist()
        candidate_movies = list(set(self.movies_df['movie_id']) - set(user_history['movie_id']))
        
        cf_weight = interaction_count / (interaction_count + self.c_weight)
        cb_weight = 1.0 - cf_weight
        
        predictions = []
        for movie_id in candidate_movies:
            cf_s = self.cf_model.predict_score(user_id, movie_id)
            cb_s = self.cb_model.predict_score(liked_movies, movie_id)
            predictions.append({
                'movie_id': movie_id,
                'cf_score': cf_s,
                'content_score': cb_s,
                'cf_weight': cf_weight,
                'final_score': (cf_weight * cf_s) + (cb_weight * cb_s),
                'method': 'hybrid'
            })
            
        preds_df = pd.DataFrame(predictions).sort_values(by='final_score', ascending=False).head(n)
        return pd.merge(preds_df, self.movies_df[['movie_id', 'title', 'genres']], on='movie_id')

hybrid_recommender = HybridRecommender(cf_model, cb_model, movies)
"""),

    ("markdown", "## 8. Demonstrations"),
    ("code", """print("--- WARM USER ---")
display(hybrid_recommender.recommend(warm_users[0], train_ratings, n=5)[['title', 'genres', 'cf_weight', 'final_score']])

print("\\n--- COLD USER (Synthetic: 2 interactions) ---")
# Create synthetic cold user
synthetic_train = pd.concat([train_ratings, ratings.head(2).assign(user_id=99999)])
display(hybrid_recommender.recommend(99999, synthetic_train, n=5)[['title', 'genres', 'cf_weight', 'final_score']])

print("\\n--- BRAND NEW USER (0 interactions) ---")
display(hybrid_recommender.recommend(88888, train_ratings, n=5)[['title', 'genres', 'method', 'final_score']])
"""),

    ("markdown", "## 9. Evaluation Metrics (Precision, Recall, NDCG)"),
    ("code", """def evaluate(user_list, n_users=20):
    precisions, recalls, ndcgs = [], [], []
    for u in user_list[:n_users]:
        relevant = set(test_ratings[(test_ratings['user_id'] == u) & (test_ratings['rating'] >= 4)]['movie_id'])
        if not relevant: continue
            
        recs = hybrid_recommender.recommend(u, train_ratings, n=10)['movie_id'].tolist()
        hits = len(set(recs).intersection(relevant))
        
        precisions.append(hits / 10)
        recalls.append(hits / len(relevant))
        
        dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recs) if r in relevant)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
        
    return {"Precision@10": np.mean(precisions), "Recall@10": np.mean(recalls), "NDCG@10": np.mean(ndcgs)}

print("Evaluating Warm Users...")
print(evaluate(warm_users, n_users=20))
""")
]

for c_type, c_source in cells:
    notebook["cells"].append(create_cell(c_type, c_source))

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/standalone_colab_recommender.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Standalone notebook generated successfully!")
