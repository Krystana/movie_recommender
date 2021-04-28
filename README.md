# movie_recommender
movie recommender project, spiced academy week 10 

A Flask web interface for a movie recommender. The user inputs the titles of 3 movies and rates them from 1-5 where 5 is best. 
The user input is then processed and a recommendation for a movie made with a loaded NMF model (non-negative matrix factorization) using the movielens dataset on a postgres server. If the model can't be loaded, a new model is trained. 

This is the first time ever I used html, please keep that in mind ;) 

future work: 
- improving input possibilities (input autocomplete to avoid mismatching duplicate movie titles from different years)
- improve model performance
- use cosine similarity instead of NMF
- web interface design/html scripts


