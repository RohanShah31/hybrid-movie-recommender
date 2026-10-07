# Hybrid Movie Recommendation System with Cold-Start Handling

## 1. Project Overview
This project is an academic implementation of a Hybrid Movie Recommendation System designed specifically to handle the "cold-start" problem. It combines Collaborative Filtering (using SVD) and Content-Based Filtering (using Genre TF-IDF) with a dynamic weighting strategy.

## 2. Problem Statement
Standard collaborative filtering systems struggle to recommend movies to new users (who have no rating history) or recommend new movies (which have no ratings). This system solves this by blending user behavior (collaborative) with movie metadata (content) depending on how much data is available for a given user.

## 3. Objectives
- Implement Matrix Factorization (SVD) for Collaborative Filtering.
- Implement Genre-based Content Filtering using TF-IDF and Cosine Similarity.
- Build a dynamic hybrid weighting strategy based on interaction counts.
- Evaluate the system on Warm users vs. Cold-Start users.
- Provide a simple popularity baseline for zero-interaction users.
- Produce understandable, student-level code.

## 4. Dataset
**MovieLens 1M** is used for this project because it is reliable, fast to process locally, and well-understood in academia. It contains 1 million ratings from 6,000 users on 4,000 movies.
- `ratings.dat`
- `movies.dat`
- `users.dat`

## 5. Technologies Used
- **Python 3.x**
- **pandas** (Data manipulation)
- **numpy** (Numerical operations)
- **scikit-learn** (TF-IDF, Cosine Similarity, Metrics)
- **scikit-surprise** (SVD algorithm for Matrix Factorization)
- **matplotlib & seaborn** (Visualizations)

## 6. Project Architecture
Data is preprocessed and split into training (80%) and testing (20%) sets. A pure collaborative model (SVD) and a content model (Genre similarity) are built. During recommendation generation, their predictions are combined dynamically.

## 7. Collaborative Filtering
Uses **Singular Value Decomposition (SVD)** from the Surprise library with 50 latent factors. It learns underlying patterns in user preferences from the rating matrix.

## 8. Content-Based Filtering
Uses **TF-IDF vectorization** on movie genres, followed by **Cosine Similarity**. If a user likes a movie, the system identifies and recommends other movies with similar genres.

## 9. Hybrid Strategy
Instead of a fixed 50/50 blend, the system calculates weight dynamically:
- `CF_Weight = interaction_count / (interaction_count + 5)`
- `Content_Weight = 1 - CF_Weight`
More interactions = higher trust in Collaborative Filtering.

## 10. Cold-Start Strategy
- **Cold User (< 5 interactions):** Content-based filtering is heavily weighted.
- **Zero-Interaction User:** System falls back to a purely **Popularity-based** approach.
- **Cold Item:** Recommended solely based on genre similarity.

## 11. Evaluation Metrics
We focus on ranking metrics rather than just RMSE:
- **Precision@10:** Percentage of top 10 recommended movies that are relevant.
- **Recall@10:** Percentage of the user's liked movies that we managed to recommend in the top 10.
- **NDCG@10:** Measures how well the relevant movies are ranked (higher positions = better score).
*Note: A relevant movie is defined as one rated >= 4.*

## 12. Results
The hybrid approach sacrifices a small amount of accuracy for "Warm" users to achieve vastly better recommendations for "Cold" users compared to pure collaborative filtering.

## 13. Failure Cases
- Movies with overly broad genres (e.g., just "Drama") degrade content-based recommendations.
- Users with highly contradictory tastes confuse both models.
- "Popularity" recommendations can dominate new-user experiences, failing to offer personalized niche content initially.

## 14. How to Run
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Download dataset: `python setup_data.py`
5. Open Jupyter: `jupyter notebook`
6. Run `notebooks/hybrid_recommender.ipynb`

## 15. Project Structure
```
hybrid-movie-recommender/
│
├── data/                      # MovieLens 1M dataset goes here
├── notebooks/
│   └── hybrid_recommender.ipynb
├── src/                       # Python modules for the core logic
│   ├── data_preprocessing.py
│   ├── collaborative_filter.py
│   ├── content_based.py
│   ├── hybrid_model.py
│   └── evaluation.py
├── results/                   # Evaluation results
├── README.md
├── WRITEUP.md                 # 2-page academic report
├── VIVA_QUESTIONS.md          # Q&A for viva preparation
├── requirements.txt
└── setup_data.py              # Script to download the dataset
```

## 16. Limitations
The content-based model is limited to genres. Integrating tags, directors, and cast would improve accuracy but increase complexity.

## 17. Future Improvements
- Include user demographics (age, gender, occupation) in a deeper hybrid model.
- Use advanced deep learning embeddings for content representation.
- Collect implicit feedback (clicks, views) to supplement explicit ratings.

## 18. Author
Created for a 2nd-Year CSE Academic Project.
