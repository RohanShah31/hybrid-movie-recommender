import numpy as np

def get_relevant_items(user_id, test_ratings, threshold=4.0):
    """
    Returns a set of movie IDs that the user liked in the test set.
    """
    user_test = test_ratings[test_ratings['user_id'] == user_id]
    liked = user_test[user_test['rating'] >= threshold]['movie_id'].tolist()
    return set(liked)

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Of the K recommended items, how many were relevant?
    """
    if k == 0: return 0.0
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))
    return hits / k

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Of all relevant items, how many did we recommend in top K?
    """
    if not relevant_items:
        return 0.0
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))
    return hits / len(relevant_items)

def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Measures ranking quality. Higher if relevant items are at the top.
    """
    if not relevant_items:
        return 0.0
        
    rec_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant_items:
            # log2(i + 2) because i is 0-indexed, and formula uses log2(rank + 1)
            dcg += 1.0 / np.log2(i + 2)
            
    # Ideal DCG: if all relevant items were at the very top
    idcg = 0.0
    num_ideal = min(len(relevant_items), k)
    for i in range(num_ideal):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0
