import pandas as pd
import os

def load_data(data_dir="data"):
    """
    Loads MovieLens 1M dataset.
    """
    movies_path = os.path.join(data_dir, "movies.dat")
    ratings_path = os.path.join(data_dir, "ratings.dat")
    
    # Load movies
    # MovieLens 1M uses '::' as separator
    movies = pd.read_csv(
        movies_path, 
        sep='::', 
        engine='python', 
        encoding='latin-1',
        names=['movie_id', 'title', 'genres']
    )
    
    # Load ratings
    ratings = pd.read_csv(
        ratings_path, 
        sep='::', 
        engine='python', 
        encoding='latin-1',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    return movies, ratings

def preprocess_data(movies, ratings):
    """
    Cleans data and calculates interaction counts.
    """
    # Calculate user interaction count
    user_counts = ratings.groupby('user_id').size().reset_index(name='user_interaction_count')
    ratings = pd.merge(ratings, user_counts, on='user_id')
    
    # Calculate movie interaction count
    movie_counts = ratings.groupby('movie_id').size().reset_index(name='movie_interaction_count')
    ratings = pd.merge(ratings, movie_counts, on='movie_id')
    
    # Calculate popularity baseline (average rating + count)
    popularity = ratings.groupby('movie_id').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    # Add popularity info to movies dataframe
    movies = pd.merge(movies, popularity, on='movie_id', how='left')
    movies['num_ratings'] = movies['num_ratings'].fillna(0)
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    
    return movies, ratings

def get_train_test_split(ratings, test_size=0.2, random_state=42):
    """
    Splits ratings into training and testing sets.
    """
    from sklearn.model_selection import train_test_split
    
    train_ratings, test_ratings = train_test_split(
        ratings, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return train_ratings, test_ratings
