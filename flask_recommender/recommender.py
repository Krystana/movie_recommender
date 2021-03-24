import pandas as pd
import numpy as np
import random
from joblib import load
from joblib import load, dump
#! pip install fuzzywuzzy
from fuzzywuzzy import fuzz

def format_dict(user_dict):
    """reformats user input dict to format movie:rating, returns dict"""
    user_dict_new = {user_dict.get('movie1'): user_dict.get('rating1'), user_dict.get('movie2'): user_dict.get('rating2'),user_dict.get('movie3'): user_dict.get('rating3')}

    return user_dict_new

def user_movie_index(user_dict, movie_names_dict):
    """compares movie titles in user input dict to titles in movie names dict with fuzzybuzzy. If sort ratio > 70 -> original
       movie title and movie Id are appended to list user_movie_index. If doubles occur, error is thrown.
        Returns list with tuples of movieId and movie title"""
    user_movie_index = []
    # for loop comparing movie names with fuzzywuzzy
    for movie in user_dict: 
        movie = str(movie).lower()
        print(' movies in user_dict: ' + movie)
        for index, moviename in movie_names_dict.items(): 
            if fuzz.token_sort_ratio(movie, moviename) > 75:  ### changed from 70!
                print('over 70: ' + moviename)
                user_movie_index.append([index, moviename])
                print('appended: ' + str(user_movie_index))      
    # throw error in terminal if double matches of movie titles occured               
    if len(user_movie_index) != len(user_dict):
        print("sorry, this doesn't work for now, please be more precise about the year or chose another movie: " + str(dict(user_movie_index).values()))

    return user_movie_index

def to_array(user_input, user_dict, df):
    """input is user dict with movie names and ratings. Creates df out of it, formats it, joins with columns of ratings df
       and keeps only new_user row. Converts df to array, returns array in shape (1, 9724,)"""
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
    user_ratings = pd.concat([user_ratings, df], axis=0, join='outer').iloc[0]  #### probably wrong here! look at NMF.ipynb!
    # get rid of Nan's:
    user_ratings = user_ratings.fillna(df.mean().round(2))
    # to array: 
    user_array = user_ratings.to_numpy()
    # reshape: 
    user_array = user_array.reshape(1, 9724)
    
    return user_array

def get_prediction(user_array, trained_model, movie_names_list):
    """takes in user array, make prediction with NMF model, shows best prediction as movie title"""
    profile = trained_model.transform(user_array)
    Q = trained_model.components_ 
    movie_preds = np.dot(profile, Q)
    best_rating = movie_preds.argmax()    #### sth not working correctly!! look above
    movie = movie_names_list[best_rating]
    #print(movie[1]) 
    print(str(best_rating) + '******' + str(movie) )
    return movie[1]
    


#if __name__ == '__main__': 

    #movie = get_prediction(user_array, trained_model, movie_names_list)
    #print(movie)

