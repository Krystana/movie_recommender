from sqlalchemy import create_engine
from sklearn.decomposition import NMF

import pandas as pd
import numpy as np
import os 

from joblib import dump, load


HOST = 'localhost'
PORT = '5432'
DB = 'movieLens'

conn_string = f'postgres://{HOST}:{PORT}/{DB}' 

engine = create_engine(conn_string)

# NMF model: 

MODEL = NMF(n_components=20, init='random', random_state=10, max_iter=10000)


# get tables to df's: 

def table_to_df():

    ratings = pd.read_sql_query('SELECT * FROM ratings', con=engine)
    ratings = ratings.set_index('userid')
    ratings = ratings.drop('time_stamps', axis=1)
    movies = pd.read_sql_query('SELECT * FROM movies', con=engine)
    return ratings, movies


def get_mean(df):
    """get mean of ratings in ratings df to fill Nan's"""
    mean = df.mean().round(2)
    return mean

def get_sparse(df):
    """get df to sparse matrix for NMF model"""
    df_sparse = df.pivot_table(index=df.index, values='rating', columns='movieid')
    return df_sparse

def to_sparse_fillna(df, mean):
    df_matrix = get_sparse(df)
    df_filled = df_matrix.fillna(mean)
    return df_filled

# get movie names dictionary: 
def get_movie_names_dict(df):

    # dict for Movie names: 
    MOVIE_NAMES = dict(zip(df['movieid'], df['title']))
    return MOVIE_NAMES

def get_movie_names_list(df):
    #list for movie names: 
    movie_names = list(zip(df['movieid'], df['title']))
    return movie_names

# fit model: 
def model_fit(df_matrix):

    MODEL.fit(df_matrix)
    return MODEL

# save model: 
def dump_model(MODEL_fitted):
    dump(MODEL_fitted, './NMF.joblib')

# load model if not there yet: 
def load_model():

    #if 'NMF.joblib' in os.listdir('.'):
    MODEL_fitted = load('./NMF.joblib')
    #else: MODEL_fitted = model_fit(ratings)
    return MODEL_fitted


#def q_matrix(model):
   # """create q matrix for predictions"""
   # Q = MODEL_fitted.components_ 



if __name__ == '__main__' :


    ...
    

