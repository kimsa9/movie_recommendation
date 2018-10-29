# movie_recommendation
Recommendation system to predict preferences of users towards movies

**Goal 1:** develop a recommendation system using information about :

- users
- movies
- preferences of some users towards some movies

The aim is to predict preferences of users towards movies for new (user,movie) pairs.

The files used are :

- movies_metadata.csv : contains the movies
- ratings.csv : contains the movie ratings for all the users
- evaluation_ratings.csv : the pairs of (user, movie) for which a rating is expected in the submission file
Evaluation

The solution will be evaluated using root mean square prediction error.

**Goal 2 :**
For all the movies that are rated in movies_metadata.csv file, create a file that contains pairs of (movie, movie) (movies are distinct) containing three columns: Id, MovieId, MovieId and Rating.

# Entries

### movies_metadata.csv
Features

- adult: Indicates if the movie is X-Rated or Adult.
- belongs\_to\_collection: A stringified dictionary that gives information on the movie series the particular film belongs to.
- budget: The budget of the movie in dollars.
- genres: A stringified list of dictionaries that list out all the genres associated with the movie.
- homepage: The Official Homepage of the move.
- id: The ID of the move.
- imdb_id: The IMDB ID of the movie.
- original_language: The language in which the movie was originally shot in.
- original_title: The original title of the movie.
- overview: A brief blurb of the movie.
- popularity: The Popularity Score assigned by TMDB.
- poster_path: The URL of the poster image.
- production_companies: A stringified list of production companies involved with the making of the movie.
- production_countries: A stringified list of countries where the movie was shot/produced in.
- release_date: Theatrical Release Date of the movie.
- revenue: The total revenue of the movie in dollars.
- runtime: The runtime of the movie in minutes.
- spoken_languages: A stringified list of spoken languages in the film.
- status: The status of the movie (Released, To Be Released, Announced, etc.)
- tagline: The tagline of the movie.
- title: The Official Title of the movie.
- video: Indicates if there is a video present of the movie with TMDB.
- vote_average: The average rating of the movie.
- vote_count: The number of votes by users, as counted by TMDB.

# Method used

- Collaborative filtering.
This method result for the rmse was 0.80.

To build and run the collaborative filtering, use the file main.py

```
spark-submit main.py
```

1. Find the best parameters and save the model. This is computed in the script **movie_recommendation/build\_als\_model.py**.
2. Apply the model found to the evaluation dataset. The script is **movie_recommendation/apply\_als**.

This method could be improved by combining it with other method such as Singular Value Decomposition. 

- Linear regression

This method did not work very well. The result for rmse was 0.98.
This method is computer in the script **movie_recommendation/linear\_regression.py**.

- Data cleaning

Before using the linear regression, the movie dataset was cleaned and merged to the rating dataset. This is computed in the script **movie_recommendation/moviedata\_cleaning.py**.

- Movie similarity

Tryed to build a item-item similarity method using cosine similarity in the script **movie_recommendation/movie\_similarity.py**.

We could also have tried clustering algorithms on the cleaned movie dataset. 

# Installation

Can be installed using `pip install --process-dependency-links -e .`   

# References

Data exploration
- https://www.kaggle.com/rounakbanik/the-story-of-film

Collaborative filtering method
- https://github.com/jadianes/spark-movie-lens/blob/master/notebooks/building-recommender.ipynb

Movies similarity based on collaborative filtering
- https://github.com/maviator/metflix/blob/master/notebooks/similarMovies.ipynb
