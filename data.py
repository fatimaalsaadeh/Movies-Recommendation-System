# Importing the required libraries
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering
import datetime
import requests, zipfile, io
from os import path
import pandas as pd
import tqdm as tqdm
from numpy import *
from sklearn.model_selection import train_test_split
import time
<<<<<<< HEAD
import pickle

=======
import matplotlib.pyplot as plt
>>>>>>> upstream/master


# Loading the mapping data which is to map each movie Id
# in the ratings with it's title and genre
# the resulted data structure is a dictionary where the
# movie id is the key, the genre and titles are values
def load_mapping_data():
    movie_data = {}
    genres_count = {}
    actual_genres = {}
    chunk_size = 500000
    df_dtype = {
        "movieId": int,
        "title": str,
        "genres": str
    }
    cols = list(df_dtype.keys())
    for df_chunk in tqdm.tqdm(pd.read_csv('ml-latest-small/movies.csv', usecols=cols, dtype=df_dtype, chunksize=chunk_size)):
        df_chunk.shape[0]
        combine_data = [list(a) for a in
                        zip(df_chunk["movieId"].tolist(), df_chunk["title"].tolist(),
                            df_chunk["genres"].tolist())]
        for a in combine_data:
            genres = a[2].split("|")
            if(a[2] not in actual_genres):
                actual_genres[a[2]]=1
            for g in genres:
                if(g in genres_count.keys()):
                    genres_count[g]=genres_count[g]+1;
                else:
                    genres_count[g]=1
            movie_data[a[0]] = [a[1], a[2]]
    del df_chunk

    return movie_data, genres_count, actual_genres

# Loading the rating data which is around 27M records it takes around 2 minutes
# the resulted data structure us a dictionary where the
# user id is the key and all their raings are values for example for user 1 :
# 1 = {
#     [movieId,rating,timestamp],
#     [movieId,rating,timestamp],
#     [movieId,rating,timestamp],
#   }

def load_data():
    rating_data = {}
    unique_user_id = []
    chunk_size = 50000
    df_dtype = {
        "userId": int,
        "movieId": int,
        "rating": float,
        "timestamp": int,
    }
    cols = list(df_dtype.keys())
    for df_chunk in tqdm.tqdm(pd.read_csv('ml-latest-small/ratings.csv', usecols=cols, dtype=df_dtype, chunksize=chunk_size)):
        user_id = df_chunk["userId"].tolist()
        unique_user_id.extend(set(user_id))
        movie_id = df_chunk["movieId"].tolist()
        rating = df_chunk["rating"].tolist()
        timestamp = df_chunk["timestamp"].tolist()
        combine_data = [list(a) for a in zip(user_id, movie_id, rating, timestamp)]
        for a in combine_data:
            if a[0] in rating_data.keys():
                rating_data[a[0]].extend([[a[0], a[1], a[2], a[3]]])
            else:
                rating_data[a[0]] = [[a[0], a[1], a[2], a[3]]]
    del df_chunk
    
    return rating_data, unique_user_id

# Split the data into training and testing
# this processes isn't being done for the whole dataset instead it's being done
# for each user id, for each user we split their ratings 80 training and 20 testing
# the resulted training and testing datasets are including the whole original dataset

def spilt_data(rating_data, unique_user_id):
    training_data = []
    testing_data = []
    t0 = time.time()
    t1 = time.time()
    for u in unique_user_id:
        if len(rating_data[u]) == 1:
            x_test = rating_data[u]
            x_train = rating_data[u]
        else:
            x_train, x_test = train_test_split(rating_data[u], test_size=0.2)
        training_data.extend(x_train)
        testing_data.extend(x_test)
    total = t1 - t0
    print(int(total))

    return training_data, testing_data

def get_movie_title(movie_id, movie_data):
    if movie_id in movie_data.keys():
        return movie_data[movie_id][0]

def get_movie_genre(movie_id, movie_data):
    if movie_id in movie_data.keys():
        return (movie_data[movie_id][1]).split("|")



# def get_train_test_data():
#     rating_data, unique_user_id = load_data()
#     training_data, testing_data = spilt_data(rating_data, unique_user_id)
#     training_dataframe = pd.DataFrame.from_records(training_data)
#     training_dataframe.columns = ["userId","movieId","rating","timestamp"]
#     testing_dataframe = pd.DataFrame.from_records(testing_data)
#     testing_dataframe.columns= ["userId","movieId","rating","timestamp"]
    
#     return training_dataframe, testing_dataframe

def get_train_test_data(new_sample = False):
    if new_sample:
        rating_data, unique_user_id = load_data()
        training_data, testing_data = spilt_data(rating_data, unique_user_id)
        training_dataframe = pd.DataFrame.from_records(training_data)
        training_dataframe.columns = ["userId","movieId","rating","timestamp"]
        testing_dataframe = pd.DataFrame.from_records(testing_data)
        testing_dataframe.columns=["userId","movieId","rating","timestamp"]
        # df_links = pd.read_csv('ml-latest-small/links.csv')
        file = open('training_dataframe.txt', 'wb')
        pickle.dump(training_dataframe, file)
        file.close()

        file = open('testing_dataframe.txt', 'wb')
        pickle.dump(testing_dataframe, file)
        file.close()
        

    else:
        file = open('training_dataframe.txt', 'rb')
        training_dataframe = pickle.load(file)
        file.close()

        file = open('testing_dataframe.txt', 'rb')
        testing_dataframe = pickle.load(file)
        file.close()

    return training_dataframe, testing_dataframe

def plots():
    # total number of movies per genre type
    index = arange(len(list(genres_count.keys())))
    plt.bar(index, list(genres_count.values()))
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('No of Movies', fontsize=5)
    plt.xticks(index, list(genres_count.keys()), fontsize=5, rotation=30)
    plt.title('No of Movies per Genre ')
    plt.show()

    # density ratings for all movies
    total_rating_df = pd.read_csv('ml-latest-small/ratings.csv')
    total_movie_df = pd.read_csv('ml-latest-small/movies.csv')
    total_combine_df = total_rating_df.merge(total_movie_df, left_on='movieId', right_on='movieId', how='left')
    df_temp1 = total_combine_df[['movieId','rating']].groupby('movieId').mean()
    plt.hist(df_temp1.rating, density = True,edgecolor='w', label ='Total')

    # Kernel Density Estimate plot per for ratings for all movies per genre
    # the plot shows most of the genres are left-skewed except for the musical movies
    for genre in genres_count.keys():
        df_temp = total_combine_df[total_combine_df['genres']==genre][['movieId','rating']].groupby('movieId').mean()
        if len(df_temp)>10:
            df_temp.rating.plot(kind='kde', label=genre)
    plt.legend()
    plt.xlabel('Rating')
    plt.title('Rating-Density')
    plt.xlim(0,5)
    plt.show()

    total_tags_df = pd.read_csv('ml-latest-small/tags.csv')
    total_combine_df = total_tags_df.merge(total_movie_df, left_on='movieId', right_on='movieId', how='left')
    print(total_combine_df.head())

if __name__ == "__main__":
    # download http://files.grouplens.org/datasets/movielens/ml-latest-small.zip with 1M records File
    # all files should be placed inside ml-latest folder
    if not path.exists('ml-latest-small'):
        print("Downloading Files for first time use: ")
        download_file = requests.get('http://files.grouplens.org/datasets/movielens/ml-latest-small.zip')
        zipped_file = zipfile.ZipFile(io.BytesIO(download_file.content)) # having First.csv zipped file.
        zipped_file.extractall()

    print("Data Loading and Processing, Estimated Time 2 minutes :")
    rating_data, unique_user_id = load_data()

    print("Training and Testing DataSets Construction, Estimated Time 40 seconds :")
    training_data, testing_data = spilt_data(rating_data, unique_user_id)

    print("Mapping Data Processing :")
    movie_data, genres_count, actual_genres = load_mapping_data()

    plots()
    print("Movie name with id = 1 :")
    print(get_movie_title(1, movie_data))

    print("Movie genre with id = 1 :")
    print(get_movie_genre(1, movie_data))



