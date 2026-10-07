# Viva Questions and Answers

**1. What is collaborative filtering?**
Collaborative filtering is a recommendation technique that predicts what a user might like based on the preferences of similar users. It assumes that if users agreed in the past, they will agree in the future.

**2. What is content-based filtering?**
Content-based filtering recommends items similar to those a user has liked in the past, based on the item's features or metadata (like genres, directors, or keywords).

**3. What is matrix factorization?**
Matrix factorization is a class of collaborative filtering algorithms. It works by decomposing the large user-item interaction matrix into two lower-dimensional matrices representing latent (hidden) factors of users and items.

**4. What is SVD?**
SVD stands for Singular Value Decomposition. It is a mathematical technique used in matrix factorization to identify underlying patterns in the data and predict missing values (ratings).

**5. Why use SVD?**
SVD is used because it effectively handles sparse datasets (where most users haven't rated most movies) and discovers hidden relationships between users and movies, leading to accurate predictions.

**6. What is cold start?**
The cold start problem occurs when a recommendation system cannot draw meaningful inferences because it has not yet gathered sufficient information about a user or an item.

**7. What is a cold-start user?**
A cold-start user is a new user who has interacted with very few items (e.g., fewer than 5 ratings), making it hard to understand their preferences.

**8. What is a cold-start item?**
A cold-start item is a newly added item (like a new movie) that has few or no ratings, making it difficult to recommend based on past user behavior.

**9. Why does collaborative filtering fail for new users?**
Collaborative filtering relies on overlapping rating histories between users. A new user has no history, so the system cannot find similar users to base recommendations on.

**10. Why does collaborative filtering fail for new items?**
Since a new item has no ratings, the system doesn't know which types of users like it, meaning it will never be recommended by a pure collaborative filtering model.

**11. Why use genres?**
Genres provide descriptive metadata about a movie. Even if a movie has no ratings, we know its genres, which allows the content-based system to recommend it based on similarity to other movies.

**12. What is TF-IDF?**
TF-IDF (Term Frequency-Inverse Document Frequency) is a technique used to convert text (like genres) into numerical vectors. It gives more weight to unique genres and less weight to very common ones.

**13. What is cosine similarity?**
Cosine similarity is a metric that measures how similar two mathematical vectors are, regardless of their size. We use it to measure how similar two movies are based on their TF-IDF genre vectors.

**14. What is hybrid recommendation?**
A hybrid recommendation system combines two or more recommendation techniques (like collaborative and content-based filtering) to improve overall performance and overcome the weaknesses of individual models.

**15. Why not use a fixed 50/50 weight?**
A fixed weight doesn't solve the cold-start problem. If a user is new, collaborative filtering is inaccurate, so relying on it for 50% of the score hurts recommendations. The weight should adapt to the user's history.

**16. How is the dynamic weight calculated?**
The dynamic weight for collaborative filtering (CF) is calculated as: `interaction_count / (interaction_count + C)`, where C is a constant (e.g., 5). The content weight is `1 - CF_weight`.

**17. Why does CF weight increase with interactions?**
As a user rates more movies, the collaborative filtering model gathers more data and becomes more accurate and reliable. Therefore, we trust its predictions more and give it a higher weight.

**18. What happens when interaction count is zero?**
When interaction count is zero, the CF weight becomes zero. The system must rely entirely on other methods, such as falling back to a popularity-based model.

**19. Why use popularity fallback?**
For completely new users, we have zero information about their personal tastes. The safest bet is to recommend popular, highly-rated movies that appeal to the general public until we learn their preferences.

**20. What is Precision@10?**
Precision@10 measures the proportion of relevant movies out of the top 10 recommended movies. (e.g., If 4 out of the 10 recommended movies were actually liked by the user, Precision@10 is 40%).

**21. What is Recall@10?**
Recall@10 measures how many of the movies the user actually liked were successfully captured in our top 10 recommendations.

**22. What is NDCG@10?**
NDCG (Normalized Discounted Cumulative Gain) measures ranking quality. It gives a higher score if the relevant movies are placed at the very top of the recommendation list rather than at the bottom.

**23. Why not use only RMSE?**
RMSE (Root Mean Square Error) only evaluates how well we predict a specific rating. In real life, predicting the exact rating matters less than correctly ranking the top 10 best movies to show the user.

**24. How did you define a relevant movie?**
We defined a relevant movie as one that the user rated 4.0 or 5.0. This assumes the user actually liked the movie and would consider it a good recommendation.

**25. How did you create the cold-start slice?**
We split the data and identified users in the test set who had fewer than 5 ratings in the training set. We evaluated their recommendations separately from the "warm" users.

**26. What are the limitations?**
Our content-based model only uses genres. Genres can be too broad (e.g., just "Drama"). We don't have tags, cast, or director information, which would make similarity matching much better.

**27. What are the failure cases?**
The system might fail for users with highly unusual or contradictory tastes, or for movies with overly generic genre tags, making content-based recommendations poor.

**28. How could this system be improved?**
We could improve it by incorporating more metadata (cast, director, plot keywords), using implicit feedback (clicks, watch time), or exploring deep learning methods for better feature extraction in the future.
