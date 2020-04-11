import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy

# ratings matrix
df_ratings = pd.read_csv('ml-latest/ratings.csv')
# print(df_ratings.head())

# print(df_ratings.rating.value_counts())

data = df_ratings.iloc[:, :-1]
print(data.head())

# load dataset
print("Reading data")
reader = Reader()
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)


# split dataset into train and test
trainset, testset = train_test_split(data, test_size=0.20)

print("Running the algorithm")
algo = SVD()
algo.fit(trainset)

print("Predictions")
predictions = algo.test(testset)


# evaluation
print("accuracy = ", accuracy.rmse(predictions))
