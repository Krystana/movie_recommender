'''main script to run flask application of movie recommender. 
Creates movie recommendation out of user input of 3 movies and its ratings from 1-5.'''

from flask import Flask, render_template, make_response, jsonify, request
from recommender import user_movie_index, to_array, get_prediction, format_dict
from model import load_model, get_movie_names_dict, get_movie_names_list, table_to_df, get_mean, get_sparse, to_sparse_fillna, model_fit, dump_model
import os

# initiate Flask
app = Flask(__name__)

@app.route('/recommender') 
def recommender():
    # get tables from db:
    ratings, movies = table_to_df()
    # ratings table to correct format: 
    ratings = get_sparse(ratings)

    # check if model is loaded, if not fit on process df ratings and fit model on it: 
    if 'NMF.joblib' in os.listdir('.'):
        trained_model = load_model()    
    else: 
        mean = get_mean(ratings) # get mean for fillna
        ratings = to_sparse_fillna(ratings, mean) # fill nans with mean
        trained_model = model_fit(ratings) # fit on processed df
        dump_model(trained_model) # save to disk

    # create list and dict of movie titles/movieIDs:
    movie_names_dict = get_movie_names_dict(movies)
    movie_names_list = get_movie_names_list(movies)

    # dic from user input: 
    user_dict = dict(request.args) 
    user_dict = format_dict(user_dict)
    print(user_dict)

    # process user input to return tuples list with movie ID and original title: 
    user_input = user_movie_index(user_dict, movie_names_dict) 
    print('***** *' + str(user_input))

    # user_input to df, join with columns of processed ratings df, fill nans and to array with size (1, 9724) for model input: 
    user_array = to_array(user_input, user_dict, ratings)

    # get predictions: 
    movie = get_prediction(user_array, trained_model, movie_names_list)
    
    # make prediciton
    return render_template('recommendation.html', 
    movie=movie, message='Hello World')

@app.route('/')
def main_page():
    return render_template('main_page.html')

if __name__ == '__main__' :

    app.run(debug=True, port=5000)

