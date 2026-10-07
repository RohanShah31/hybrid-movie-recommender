from surprise import SVD, Dataset, Reader
import pandas as pd

class CollaborativeFilteringSVD:
    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = SVD(n_factors=self.n_factors, random_state=self.random_state)
        self.trainset = None

    def fit(self, train_ratings):
        """
        Trains the SVD model using the training ratings.
        """
        # Surprise requires a specific format
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_ratings[['user_id', 'movie_id', 'rating']], 
            reader
        )
        
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict_score(self, user_id, movie_id):
        """
        Predicts a rating for a user-movie pair.
        Returns a score between 1 and 5.
        """
        # If the user or movie is completely unknown to the model (0 interactions in train)
        # SVD will just return the global average rating.
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
