import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, cf_model, cb_model, movies_df, c_weight=5):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.movies_df = movies_df
        self.c_weight = c_weight
        
    def calculate_weights(self, interaction_count):
        """
        Dynamically calculates weights. 
        More interactions = higher CF weight.
        """
        cf_weight = interaction_count / (interaction_count + self.c_weight)
        content_weight = 1.0 - cf_weight
        return cf_weight, content_weight

    def recommend(self, user_id, train_ratings, n=10):
        """
        Generates Top-N recommendations for a user.
        """
        # 1. Get user's interaction count and past movies
        user_history = train_ratings[train_ratings['user_id'] == user_id]
        interaction_count = len(user_history)
        
        # New User / Zero Interaction handling (Fallback to Popularity)
        if interaction_count == 0:
            print(f"User {user_id} is a completely new user. Using Popularity Fallback.")
            popular = self.movies_df.sort_values(
                by=['num_ratings', 'avg_rating'], 
                ascending=[False, False]
            ).head(n)
            
            results = []
            for _, row in popular.iterrows():
                results.append({
                    'movie_id': row['movie_id'],
                    'title': row['title'],
                    'genres': row['genres'],
                    'cf_score': 0,
                    'content_score': 0,
                    'cf_weight': 0,
                    'content_weight': 0,
                    'final_score': row['avg_rating'],
                    'method': 'popularity'
                })
            return pd.DataFrame(results)

        # 2. Get movies user has liked (rating >= 4) for content-based profile
        liked_movies = user_history[user_history['rating'] >= 4]['movie_id'].tolist()
        
        # 3. Create candidate pool (all movies user hasn't seen)
        watched_movies = set(user_history['movie_id'])
        all_movies = set(self.movies_df['movie_id'])
        candidate_movies = list(all_movies - watched_movies)
        
        # 4. Calculate dynamic weights
        cf_weight, cb_weight = self.calculate_weights(interaction_count)
        
        predictions = []
        for movie_id in candidate_movies:
            # Get Collaborative Score
            cf_score = self.cf_model.predict_score(user_id, movie_id)
            
            # Get Content-Based Score
            cb_score = self.cb_model.predict_score(liked_movies, movie_id)
            
            # Combine Scores
            final_score = (cf_weight * cf_score) + (cb_weight * cb_score)
            
            predictions.append({
                'movie_id': movie_id,
                'cf_score': cf_score,
                'content_score': cb_score,
                'cf_weight': cf_weight,
                'content_weight': cb_weight,
                'final_score': final_score,
                'method': 'hybrid'
            })
            
        # 5. Sort and get Top N
        predictions_df = pd.DataFrame(predictions)
        top_predictions = predictions_df.sort_values(by='final_score', ascending=False).head(n)
        
        # Merge with movie metadata
        final_recs = pd.merge(top_predictions, self.movies_df[['movie_id', 'title', 'genres']], on='movie_id')
        
        return final_recs
