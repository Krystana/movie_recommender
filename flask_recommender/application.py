import os

from flask import Flask, render_template, make_response, jsonify, request
from recommender import get_recommendation, user_movie_index, to_array, get_prediction, format_dict
from model import load_model, get_movie_names_dict, get_movie_names_list, table_to_df, get_mean, get_sparse, to_sparse_fillna, model_fit

app = Flask(__name__)

@app.route('/recommender') # <-- decorater #app.route: define which url on webpage work
def recommender():

    # get tables from db:
    ratings, movies = table_to_df()
    # ratings table to correct format: 
    ratings = get_sparse(ratings)


    # check if model is loaded, if not fit on process df ratings and fit model on it: 
    if 'NMF.joblib' in os.listdir('.'):
        trained_model = load_model()    # load model form model.py
    else: 
        mean = get_mean(ratings) # get mean for fillna
        #ratings = get_sparse(ratings) # to sparse df
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
    
    #movie = get_recommendation(user_input)
    # make prediciton
    return render_template('recommendation.html', 
    movie=movie, message='Hello World')

@app.route('/')
def main_page():
    return render_template('main_page.html')

#@app.route('/all')
#def all_movies():
    #data = {'movies': MOVIES}
    #return make_response(jsonify(data))

if __name__ == '__main__' :
# this block is run when we run application, not when we import it equals def main
    app.run(debug=True, port=5000)

