import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering, KNNBasic
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

def compute_error(actual_ratings, estimate_ratings):
	ratings = np.array(actual_ratings)
	estimate = np.array(estimate_ratings)

	rmse = np.sqrt(np.sum(np.square(np.subtract(ratings, estimate)))/np.size(ratings))
	mae = np.sum(np.abs(np.subtract(ratings, estimate)))/np.size(ratings)

	return rmse, mae

def svdalgorithm(trainset, testset):

	print("\n" + "-" *5 + " SVD algorithm using surprise package " + "-" *5)
	algo = SVD()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions


def baseline(trainset, testset):

	print("\n" + "-" *5 + " Baseline algorithm using surprise package " + "-" *5)
	algo = BaselineOnly()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions

def svdpp(trainset, testset):
	# Matrix factorization - SVD++
	print("\n" + "-" *5 + " SVD++ algorithm using surprise package " + "-" *5)
	algo = SVDpp()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions

def slopeOne(trainset, testset):
	# Slope One
	print("\n" + "-" *5 + " SlopeOne algorithm using surprise package " + "-" *5)
	algo = SlopeOne()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions

def coClustering(trainset, testset):
	# CoClustering
	print("\n" + "-" *5 + " CoClustering algorithm using surprise package " + "-" *5)
	algo = CoClustering()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions

def kNNBasic(trainset, testset):
	# KNN basic
	print("\n" + "-" *5 + " KNNBasic algorithm using surprise package " + "-" *5)
	sim_options = {
	                'name': 'MSD',      # MSD similarity measure gives the best result
	              #  'user_based': True  # compute  similarities between users: MAE = 0.7744112391896695
	               'user_based': False  # compute  similarities between items: MAE = 0.7685376263051
	               }
	algo = KNNBasic(sim_options = sim_options)
	# algo = KNNBasic()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions

def hybrid(trainset, testset):
	prediction_baseline = baseline(trainset, testset)
	predictions_svd = svdalgorithm(trainset, testset)

	baseline_estimate = []
	actual_ratings = []
	svd_estimate = []

	for p in prediction_baseline:
		baseline_estimate.append(p[3])
		actual_ratings.append(p[2])

	for p in predictions_svd:
		svd_estimate.append(p[3])

	hybrid_estimate = np.multiply(baseline_estimate, 0.1) + np.multiply(svd_estimate, 0.9)
	rmse, mae = compute_error(actual_ratings, hybrid_estimate)

	print("\n" + "-" *5 + " Hybrid algorithm " + "-" *5)
	print("RMSE: ", rmse)
	print("MAE: ", mae)

if __name__ == "__main__":

	df_train, df_test = data.get_train_test_data(new_sample = False)
	trainset, testset = convert_traintest_dataframe_forsurprise(df_train, df_test)

	# baseline(trainset, testset)
	# svdalgorithm(trainset, testset)
	# svdpp(trainset, testset)
	# slopeOne(trainset, testset)
	# coClustering(trainset, testset)
	# kNNBasic(trainset, testset)

	hybrid(trainset, testset)