import pandas as pd
import numpy as np
import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.abspath('src'))

from data_preprocessing import load_data, preprocess_data, get_train_test_split
from collaborative_filter import CollaborativeFilteringSVD
from content_based import ContentBasedFiltering
from hybrid_model import HybridRecommender
from evaluation import get_relevant_items, precision_at_k, recall_at_k, ndcg_at_k

def main():
    print("="*50)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM DEMO")
    print("="*50)
    
    # 1. Load Data
    print("\n[1/5] Loading and Preprocessing Data...")
    raw_movies, raw_ratings = load_data(os.path.join('data', 'ml-1m'))
    movies, ratings = preprocess_data(raw_movies, raw_ratings)
    print(f"Loaded {len(ratings)} ratings across {movies['movie_id'].nunique()} movies.")

    # 2. Split Data
    print("\n[2/5] Creating Train/Test Split...")
    train_ratings, test_ratings = get_train_test_split(ratings, test_size=0.2)
    
    train_user_counts = train_ratings['user_id'].value_counts()
    warm_users = train_user_counts[train_user_counts >= 5].index.tolist()
    
    # Create a synthetic cold user for demonstration
    example_cold_user = 99999
    synthetic_history = ratings.head(2).copy()
    synthetic_history['user_id'] = example_cold_user
    train_ratings = pd.concat([train_ratings, synthetic_history])

    # 3. Train Models
    print("\n[3/5] Training Collaborative Filter (SVD)...")
    cf_model = CollaborativeFilteringSVD(n_factors=50)
    cf_model.fit(train_ratings)
    
    print("\n[4/5] Training Content-Based Filter (TF-IDF)...")
    cb_model = ContentBasedFiltering()
    cb_model.fit(movies)
    
    # 4. Initialize Hybrid
    hybrid_recommender = HybridRecommender(cf_model, cb_model, movies, c_weight=5)
    
    # 5. Demonstrations
    print("\n" + "="*50)
    print("DEMO 1: WARM USER (Rich History)")
    print("="*50)
    example_warm_user = warm_users[0]
    print(f"Generating recommendations for Warm User {example_warm_user}...")
    recs = hybrid_recommender.recommend(example_warm_user, train_ratings, n=5)
    print(recs[['title', 'genres', 'cf_weight', 'final_score']].to_string(index=False))
    
    print("\n" + "="*50)
    print("DEMO 2: COLD USER (2 Ratings)")
    print("="*50)
    print(f"Generating recommendations for Cold User {example_cold_user}...")
    recs = hybrid_recommender.recommend(example_cold_user, train_ratings, n=5)
    print(recs[['title', 'genres', 'cf_weight', 'final_score']].to_string(index=False))
    
    print("\n" + "="*50)
    print("DEMO 3: NEW USER (0 Ratings - Fallback)")
    print("="*50)
    print(f"Generating recommendations for completely new User 88888...")
    recs = hybrid_recommender.recommend(88888, train_ratings, n=5)
    print(recs[['title', 'genres', 'method', 'final_score']].to_string(index=False))
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
