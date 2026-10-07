from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class ContentBasedFiltering:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
        self.similarity_matrix = None
        self.movie_indices = None
        self.movies_df = None

    def fit(self, movies_df):
        """
        Creates TF-IDF vectors for movie genres and calculates cosine similarity.
        """
        self.movies_df = movies_df.copy()
        # Create a mapping from movie_id to matrix index
        self.movie_indices = pd.Series(
            self.movies_df.index, 
            index=self.movies_df['movie_id']
        ).to_dict()
        
        # Fit TF-IDF on genres (e.g., Action|Comedy)
        tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['genres'])
        
        # Calculate cosine similarity between all movies
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_similar_movies(self, movie_id, top_n=10):
        """
        Returns a list of similar movies and their similarity scores.
        """
        if movie_id not in self.movie_indices:
            return []
            
        idx = self.movie_indices[movie_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the movie itself
        sim_scores = sim_scores[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df['movie_id'].iloc[movie_indices].tolist()

    def predict_score(self, user_liked_movies, target_movie_id):
        """
        Predicts a score for a target movie based on its similarity 
        to movies the user has already liked.
        """
        if target_movie_id not in self.movie_indices:
            return 0.0
            
        target_idx = self.movie_indices[target_movie_id]
        
        similarities = []
        for liked_movie_id in user_liked_movies:
            if liked_movie_id in self.movie_indices:
                liked_idx = self.movie_indices[liked_movie_id]
                sim = self.similarity_matrix[target_idx][liked_idx]
                similarities.append(sim)
                
        if not similarities:
            return 0.0
            
        # Score is between 0 and 1. We scale it to 1-5 to match CF.
        avg_sim = np.mean(similarities)
        scaled_score = 1 + (avg_sim * 4)
        return scaled_score
