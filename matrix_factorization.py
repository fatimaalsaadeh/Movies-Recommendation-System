import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering
from surprise import SVD, accuracy
import data


def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    training_dataframe = training_dataframe.iloc[:, :-1]
    testing_dataframe = testing_dataframe.iloc[:, :-1]
    reader = Reader(rating_scale=(0,5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset=testset.construct_testset(testset.raw_ratings)
    return([trainset,testset])

def svdalgorithm(trainset, testset):
    algo = SVD()
    algo.fit(trainset)
    print("Predictions")
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

def baseline(trainset, testset):
    algo = BaselineOnly()
    algo.fit(trainset)
    print("Predictions")
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)


if __name__ == "__main__":

	df_train, df_test = data.get_train_test_data()
	trainset, testset = convert_traintest_dataframe_forsurprise(df_train, df_test)
	print("Baseline algorithm using surprise package")
	baseline(trainset, testset)
	print("SVD algorithm using surprise package")
	svdalgorithm(trainset,testset)
