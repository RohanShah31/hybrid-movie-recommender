# Project Write-Up: Hybrid Movie Recommendation System with Cold-Start Handling

## Abstract
This project implements a hybrid movie recommendation system designed to handle the "cold-start" problem commonly encountered in real-world streaming platforms. The cold-start problem occurs when a system needs to recommend items to new users with little or no interaction history, or when it needs to recommend newly added items. By combining collaborative filtering (using Singular Value Decomposition) and content-based filtering (using TF-IDF on movie genres), this system dynamically balances the weights of both approaches based on the user's interaction count. The results show that our hybrid approach effectively maintains recommendation quality across both warm (experienced) and cold (new) users.

## Problem Statement
Standard recommendation engines rely heavily on collaborative filtering, which assumes users with similar past behavior will have similar future preferences. However, this approach fails for new users (no past behavior) and new items (no ratings). This project aims to build a recommendation system that overcomes this limitation by using movie metadata (genres) to supplement predictions when collaborative data is sparse.

## Methodology
The system uses the MovieLens 1M dataset, which contains 1 million ratings from 6,000 users on 4,000 movies. We defined a "cold user" as someone with fewer than 5 ratings in our training set, and a "warm user" as someone with 5 or more ratings.

### Collaborative Filtering
We implemented Matrix Factorization using Singular Value Decomposition (SVD). The basic idea is to decompose the large User-Item rating matrix into two smaller matrices: user latent factors and item latent factors. This helps the system discover hidden preferences. We kept the number of latent factors moderate (50 factors) to avoid overfitting and keep the model interpretable.

### Content-Based Filtering
For the content-based component, we utilized the movie genres. We converted the genre text (e.g., "Animation|Children's|Comedy") into vectors using TF-IDF (Term Frequency-Inverse Document Frequency). We then calculated the Cosine Similarity between these vectors. If a user likes action movies, the content-based system will find and recommend other action movies based on genre similarity.

### Hybrid Approach with Dynamic Weighting
Instead of a static 50/50 split, our hybrid model dynamically adjusts weights based on the user's interaction count.
`CF_Weight = interaction_count / (interaction_count + 5)`
`Content_Weight = 1 - CF_Weight`
For a user with 0 interactions, the CF weight is 0, relying entirely on content (or a popularity baseline). For a user with 50 interactions, CF weight becomes dominant. This elegantly solves the cold-start problem.

### Cold-Start Handling
- **New Users:** If a user has 0 interactions, we fall back to a Popularity-Based recommendation (recommending top-rated movies with many reviews).
- **Cold Users (1-4 interactions):** Content-based filtering receives a higher weight to compensate for the lack of collaborative data.
- **New Items:** New movies without ratings rely entirely on their genre similarity (content-based) to be recommended.

## Evaluation
We evaluated our system by generating Top-10 recommendations for users in our test set. Our primary metrics were:
- **Precision@10:** The proportion of recommended movies that were relevant.
- **Recall@10:** The proportion of relevant movies that were successfully recommended.
- **NDCG@10:** Normalized Discounted Cumulative Gain, which measures if relevant movies are ranked near the top.
We defined a "relevant" movie as one that the user rated 4.0 or higher.

## Results
The evaluation demonstrated that while pure collaborative filtering performed best for warm users, its performance dropped significantly for cold users. The dynamic hybrid system maintained stable performance across both groups, sacrificing a small amount of accuracy on warm users to achieve massive gains for cold users.

## Failure Analysis
While the system performs well, we observed some failure cases:
1. **Overly Broad Genres:** Some movies only have the genre "Drama", making it hard for the content-based system to find meaningful similarities.
2. **Unusual User Preferences:** If a user rated one horror movie highly and one children's movie highly, the content-based system struggles to find a consistent pattern compared to collaborative filtering.

## Conclusion
This project successfully built an understandable, dynamic hybrid recommendation engine. By combining SVD with TF-IDF genre similarity, we effectively managed the cold-start problem without resorting to overly complex deep learning architectures.
