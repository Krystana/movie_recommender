'''script connects to postgres database and contains functions to convert data to dfs, 
   create list/dict of movietitles and id's, load, fit and dump model'''

from sqlalchemy import create_engine
from sklearn.decomposition import NMF
import pandas as pd
import numpy as np 
from joblib import dump, load
import os

# connection to database
HOST = 'localhost'
PORT = '5432'
DB = 'movieLens'

conn_string = f'postgres://{HOST}:{PORT}/{DB}' 
engine = create_engine(conn_string)

# create model and hyperparameters
MODEL = NMF(n_components=20, init='random', random_state=10, max_iter=10000)

def table_to_df():
    """queries tables ratings and movies from postgres database, sets index in ratings to user id  and drops timestamp
        returns 2 dataframes"""
    ratings = pd.read_sql_query('SELECT * FROM ratings', con=engine)
    ratings = ratings.set_index('userid')
    ratings = ratings.drop('time_stamps', axis=1)
    movies = pd.read_sql_query('SELECT * FROM movies', con=engine)
    return ratings, movies

def get_mean(df):
    """gets mean of dataframe, returns mean"""
    mean = df.mean().round(2)
    return mean

def get_sparse(df):
    """converts dataframe to sparse by pivoting, returns dataframe"""
    df_sparse = df.pivot_table(index=df.index, values='rating', columns='movieid')
    return df_sparse

def to_sparse_fillna(df, mean):
    """fills Nan's in sparse df with mean of df, returns dataframe"""
    df_filled = df.fillna(mean)
    return df_filled
 
def get_movie_names_dict(df):
    """creates dictionary out of movie id and title in df, returns dict"""
    MOVIE_NAMES = dict(zip(df['movieid'], df['title']))
    return MOVIE_NAMES

def get_movie_names_list(df):
    """creates list of tuples out of movie id and title in df, returns list"""
    movie_names = list(zip(df['movieid'], df['title']))
    return movie_names

def model_fit(df_matrix):
    """fit model to matrix, returns fitted model"""
    MODEL.fit(df_matrix)
    return MODEL

def dump_model(MODEL_fitted):
    """saves model to disk"""
    dump(MODEL_fitted, './NMF.joblib')

def load_model():
    """loads model from disk"""
    MODEL_fitted = load('./NMF.joblib')
    return MODEL_fitted

if __name__ == '__main__' :
    ...
    

