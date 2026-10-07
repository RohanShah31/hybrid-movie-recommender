import json
import os

def create_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
        # Remove trailing newline from last element if empty
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
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = [
    ("markdown", "# Hybrid Movie Recommendation System with Cold-Start Handling\n\nThis notebook demonstrates a hybrid recommendation engine combining Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF on genres), dynamically weighted based on user interaction count to elegantly solve the Cold-Start problem."),
    
    ("markdown", "## 1. Setup and Imports\nWe use pandas for data manipulation, scikit-learn for content similarity, and surprise for matrix factorization."),
    
    ("code", """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.abspath('../src'))

from data_preprocessing import load_data, preprocess_data, get_train_test_split
from collaborative_filter import CollaborativeFilteringSVD
from content_based import ContentBasedFiltering
from hybrid_model import HybridRecommender
from evaluation import get_relevant_items, precision_at_k, recall_at_k, ndcg_at_k

# Set plotting style
sns.set_theme(style="whitegrid")
"""),
    
    ("markdown", "## 2. Load and Preprocess Data\nWe are using the MovieLens 1M dataset. An interaction is defined as a rating. A user with fewer than 5 interactions in our training set is considered a **Cold-Start User**."),
    
    ("code", """# Ensure data is downloaded
if not os.path.exists('../data/ratings.dat'):
    print("Please run setup_data.py first to download the dataset.")
else:
    raw_movies, raw_ratings = load_data('../data')
    movies, ratings = preprocess_data(raw_movies, raw_ratings)
    
    print(f"Total Users: {ratings['user_id'].nunique()}")
    print(f"Total Movies: {movies['movie_id'].nunique()}")
    print(f"Total Ratings: {len(ratings)}")
"""),

    ("markdown", "## 3. Data Visualization\nLet's visualize the distribution of user interactions to see why the cold-start problem exists."),
    
    ("code", """plt.figure(figsize=(10, 5))
sns.histplot(ratings.groupby('user_id').size(), bins=100, kde=False)
plt.title('Distribution of User Ratings (Interaction Count)')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Number of Users')
plt.xlim(0, 500)
plt.show()
"""),

    ("markdown", "## 4. Train/Test Split\nWe will use 80% of ratings for training our models, and 20% to evaluate our recommendations. Crucially, the test data is hidden during training to prevent data leakage."),
    
    ("code", """train_ratings, test_ratings = get_train_test_split(ratings, test_size=0.2, random_state=42)
print(f"Training ratings: {len(train_ratings)}")
print(f"Testing ratings: {len(test_ratings)}")

# Define Cold-Start and Warm users based on training interactions
train_user_counts = train_ratings['user_id'].value_counts()
cold_users = train_user_counts[train_user_counts < 5].index.tolist()
warm_users = train_user_counts[train_user_counts >= 5].index.tolist()

print(f"Number of Cold-Start Users (< 5 interactions in train): {len(cold_users)}")
print(f"Number of Warm Users (>= 5 interactions in train): {len(warm_users)}")
"""),

    ("markdown", "## 5. Train Collaborative Filtering (SVD)\nSVD creates a user-item matrix and factorizes it to find hidden latent features. We use `n_factors=50` to keep the model simple and avoid overfitting."),
    
    ("code", """print("Training SVD Model...")
cf_model = CollaborativeFilteringSVD(n_factors=50)
cf_model.fit(train_ratings)
print("SVD Training Complete.")
"""),

    ("markdown", "## 6. Train Content-Based Filtering\nWe convert the pipe-separated genres into TF-IDF vectors. Then we compute the Cosine Similarity between all movies. A movie gets a high score if it is similar to movies the user has already rated highly."),
    
    ("code", """print("Building Content-Based Model...")
cb_model = ContentBasedFiltering()
cb_model.fit(movies)
print("Content-Based Model Ready.")
"""),

    ("markdown", "## 7. Initialize Hybrid Recommender\nOur hybrid recommender dynamically changes the weight of Collaborative Filtering based on how many ratings the user has.\n`CF_weight = interactions / (interactions + C)` (where C=5)\nIf a user is completely new (0 interactions), it uses a Popularity-Based fallback."),
    
    ("code", """hybrid_recommender = HybridRecommender(cf_model, cb_model, movies, c_weight=5)
"""),

    ("markdown", "## 8. Recommendation Examples\nLet's see how the system behaves for different types of users."),
    
    ("markdown", "### Example 1: A Warm User (Lots of history)"),
    ("code", """# Pick a warm user
if len(warm_users) > 0:
    example_warm_user = warm_users[0]
    print(f"Generating recommendations for Warm User {example_warm_user}")
    recs = hybrid_recommender.recommend(example_warm_user, train_ratings, n=5)
    display(recs[['title', 'genres', 'cf_score', 'content_score', 'cf_weight', 'final_score']])
"""),

    ("markdown", "### Example 2: A Cold-Start User (Little history)"),
    ("code", """# Pick a cold user (simulate one if none exist naturally in this exact 80/20 split)
if len(cold_users) > 0:
    example_cold_user = cold_users[0]
else:
    # Synthetically create a cold user by taking only 2 ratings
    example_cold_user = 99999
    synthetic_history = ratings.head(2).copy()
    synthetic_history['user_id'] = example_cold_user
    train_ratings = pd.concat([train_ratings, synthetic_history])
    
print(f"Generating recommendations for Cold User {example_cold_user}")
recs = hybrid_recommender.recommend(example_cold_user, train_ratings, n=5)
display(recs[['title', 'genres', 'cf_score', 'content_score', 'cf_weight', 'final_score']])
"""),

    ("markdown", "### Example 3: A Brand New User (Zero history)"),
    ("code", """print(f"Generating recommendations for Brand New User (ID: 88888)")
recs = hybrid_recommender.recommend(88888, train_ratings, n=5)
display(recs[['title', 'genres', 'method', 'final_score']])
"""),

    ("markdown", "## 9. Evaluation: Warm vs Cold Users\nWe evaluate the model on a sample of users using Precision@10, Recall@10, and NDCG@10."),
    
    ("code", """def evaluate_users(user_list, n_users=50):
    users_to_evaluate = user_list[:n_users]
    
    metrics = {'precision': [], 'recall': [], 'ndcg': []}
    
    for u in users_to_evaluate:
        # Get relevant items (rating >= 4) from TEST set
        relevant = get_relevant_items(u, test_ratings, threshold=4.0)
        if not relevant: continue # Skip if they liked nothing in test set
            
        # Get recommendations using TRAIN set history
        recs = hybrid_recommender.recommend(u, train_ratings, n=10)
        rec_items = recs['movie_id'].tolist()
        
        metrics['precision'].append(precision_at_k(rec_items, relevant, k=10))
        metrics['recall'].append(recall_at_k(rec_items, relevant, k=10))
        metrics['ndcg'].append(ndcg_at_k(rec_items, relevant, k=10))
        
    return {
        'Precision@10': np.mean(metrics['precision']),
        'Recall@10': np.mean(metrics['recall']),
        'NDCG@10': np.mean(metrics['ndcg'])
    }

print("Note: Evaluation takes a few minutes. Running on a small sample for demonstration.")
# Evaluate warm users
print("\\nEvaluating Warm Users...")
warm_results = evaluate_users(warm_users, n_users=20)
print(warm_results)

# Evaluate cold users (using synthetic if needed)
print("\\nEvaluating Cold Users...")
if len(cold_users) < 10:
    print("Not enough naturally occurring cold users in this split. Demonstrating concept.")
else:
    cold_results = evaluate_users(cold_users, n_users=20)
    print(cold_results)
"""),

    ("markdown", "## 10. Failure Case Analysis\nSometimes the model makes poor recommendations. Why?\n1. **Vague Genres:** Recommending a movie purely because it is tagged 'Drama' is not very personalized.\n2. **Conflicting Tastes:** If a user rated 'Toy Story' (Animation) and 'Saw' (Horror) as 5 stars, the content-based similarity gets confused and averages these out poorly.\n3. **New User Dominance:** The Popularity fallback for new users means every new user sees exactly the same 'Top 10' movies, offering zero personalization until they start rating.")
]

for c_type, c_source in cells:
    notebook["cells"].append(create_cell(c_type, c_source))

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/hybrid_recommender.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
