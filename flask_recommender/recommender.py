import pandas as pd
import numpy as np
import random
from joblib import load
from joblib import load, dump
#! pip install fuzzywuzzy
from fuzzywuzzy import fuzz

#from model import get_movie_names_dict, get_movie_names_list


# reformat user input dict to make it work in function that I wrote not checking the real format of the dict...
def format_dict(user_dict):

    user_dict_new = {user_dict.get('movie1'): user_dict.get('rating1'), user_dict.get('movie2'): user_dict.get('rating2'),user_dict.get('movie3'): user_dict.get('rating3')}
    return user_dict_new

# process user input: 
def user_movie_index(user_dict, movie_names_dict):
    """compares movie titles in user input dict to titles in movie names dict with fuzzybuzzy. If sort ration > 70 original
       movie title and movie Id are appended to list user_movie_index. If doubles occur, error is thrown.
        returns list with tuples of movieId and movie title"""
    
    user_movie_index = []

    for movie in user_dict: 
        movie = str(movie).lower()
        print(' movies in user_dict: ' + movie)
        for index, moviename in movie_names_dict.items(): 
            if fuzz.token_sort_ratio(movie, moviename) > 70:
                print('over 70: ' + moviename)
                user_movie_index.append([index, moviename])
                print('appended: ' + str(user_movie_index))
            
    if len(user_movie_index) != len(user_dict):
        print("sorry, this doesn't work for now, please be more precise about the year or chose another movie: " + str(dict(user_movie_index).values()))
   
    return user_movie_index

def to_array(user_input, user_dict, df):
    """input user dict with moive names and ratings. Creates df out of it, formats it, joins with columns of ratings df
       and keeps only new_user row. To array, returns array in shape (1, 9724,)"""
    # to df:
    user_df = pd.DataFrame(user_input)
    # add ratings from user_dict (input) to column
    user_df['rating'] = user_dict.values()
    # drop movie titles
    user_df.drop([1], axis=1, inplace=True)
    # rename Movie ID column for better overview
    user_df.rename(columns = {0:'movieId'}, inplace=True)
    # to right format:
    user_ratings = user_df.set_index(['movieId']).transpose()
    # join with ratings df and keep only 1st row
    user_ratings = pd.concat([user_ratings, df], axis=0, join='outer').iloc[0]
    # get rid of Nan's:
    user_ratings = user_ratings.fillna(df.mean().round(2))
    # to array: 
    user_array = user_ratings.to_numpy()
    # reshape: 
    user_array = user_array.reshape(1, 9724)
    
    
    
    return user_array

def get_prediction(user_array, trained_model, movie_names_list):
    """take in user array, make prediction, show best prediction movie title"""
    profile = trained_model.transform(user_array)
    Q = trained_model.components_ 
    movie_preds = np.dot(profile, Q)
    best_rating = movie_preds.argmax()
    movie = movie_names_list[best_rating]
    print(movie[1])
    
    return movie[1]
    

def get_recommendation(user_input: dict):   # take in array!


    m1 = user_input["movie1"]
    r1 = user_input["rating1"]

    m2 = user_input["movie2"]
    r2 = user_input["rating2"]

    m3 = user_input["movie3"]
    r3 = user_input["rating3"]

    """The rest is up to you....HERE IS SOME PSEUDOCODE:
    
       1. Train the model (NMF), OR the model is already pre-trained.
       2. Process the input, e.g. convert movie titles into numbers: movie title -> column numbers
       3. Data validation, e.g. spell check....
       4. Convert the user input into an array of length len(df.columns), ~9742

       --here is where the cosine similarity will be a bit different--

       5. user_profile = nmf.transform(user_array). The "hidden feature profile" of the user, e.g. (9742, 20)
       6. results = np.dot(user_profile, nmf.components_)
       7. Sort the array, map the top N values to movie names.
       8. Return the titles. 
    """

    return random.choice([m1, m2, m3])

#get_recommendation(user_dict)

if __name__ == '__main__': # still works if this is commented out py. but then this print statement is printed in terminal
    #movie = get_recommendation(user_input)
    movie = get_prediction(user_array, trained_model, movie_names_list)
    print(movie)
# put  test print code here, if condition is not triggered by import to application e.g. is not executed!
