import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering, KNNBasic
import data
from collections import defaultdict


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
				   # 'user_based': True  # compute  similarities between users: MAE = 0.7744112391896695
				   'user_based': False  # compute  similarities between items: MAE = 0.7685376263051
				   }
	algo = KNNBasic(sim_options = sim_options)
	# algo = KNNBasic()
	algo.fit(trainset)
	predictions = algo.test(testset)
	accuracy.rmse(predictions)
	accuracy.mae(predictions)
	return predictions

def movie_recommendation(predictions, n=10):
	# First map the predictions to each user.
	algorithmrecommendations_for_each_user = defaultdict(list)
	testdaterecommendations_for_each_user = defaultdict(list)
	# Creating a dictionary with user_id as the key and the movie_id and the estimated_rating as the value
	for user_id, movie_id, true_rating, estimated_rating, _ in predictions:
		algorithmrecommendations_for_each_user[user_id].append((movie_id, estimated_rating))
		testdaterecommendations_for_each_user[user_id].append((movie_id, true_rating))
	# Now we will sort the Estimated_rating of different movies for each user
	for user_id, user_ratings in algorithmrecommendations_for_each_user.items():
		user_ratings.sort(key=lambda x: x[1], reverse=True)
		# Filtering the values to top n
		algorithmrecommendations_for_each_user[user_id] = user_ratings[:n]
	# Now we will sort the True_rating of different movies for each user
	for user_id, user_ratings in testdaterecommendations_for_each_user.items():
		user_ratings.sort(key=lambda x: x[1], reverse=True)
		# Filtering the values to top n
		testdaterecommendations_for_each_user[user_id] = user_ratings[:n]
	return([algorithmrecommendations_for_each_user,testdaterecommendations_for_each_user])

def precision_recall_calculation(predictions, threshold=3.5):

	# First map the predictions to each user.
	user_predict_true = defaultdict(list)
	for user_id, movie_id, true_rating, predicted_rating, _ in predictions:
		user_predict_true[user_id].append((predicted_rating, true_rating))

	precisions = dict()
	recalls = dict()
	for user_id, user_ratings in user_predict_true.items():

		# Sort user ratings by estimated value
		user_ratings.sort(key=lambda x: x[0], reverse=True)

		# Number of relevant items
		no_of_relevant_items = sum((true_rating >= threshold) for (predicted_rating, true_rating) in user_ratings)

		# Number of recommended items in top 10
		no_of_recommended_items = sum((predicted_rating >= threshold) for (predicted_rating, true_rating) in user_ratings[:10])

		# Number of relevant and recommended items in top 10
		no_of_relevant_and_recommended_items = sum(((true_rating >= threshold) and (predicted_rating >= threshold)) for (predicted_rating, true_rating) in user_ratings[:10])

		# Precision: Proportion of recommended items that are relevant
		precisions[user_id] = no_of_relevant_and_recommended_items / no_of_recommended_items if no_of_recommended_items != 0 else 1

		# Recall: Proportion of relevant items that are recommended
		recalls[user_id] = no_of_relevant_and_recommended_items / no_of_relevant_items if no_of_relevant_items != 0 else 1

	# Averaging the values for all users
	average_precision=sum(precision for precision in precisions.values()) / len(precisions)
	average_recall=sum(recall for recall in recalls.values()) / len(recalls)
	F_score=(2*average_precision*average_recall) / (average_precision + average_recall)
	
	return [average_precision, average_recall, F_score]

def hybrid(trainset, testset):
	prediction_baseline = baseline(trainset, testset)
	predictions_svd = svdpp(trainset, testset)

	baseline_estimate = []
	actual_ratings = []
	svd_estimate = []

	for p in prediction_baseline:
		baseline_estimate.append(p[3])
		actual_ratings.append(p[2])

	for p in predictions_svd:
		svd_estimate.append(p[3])

	hybrid_estimate = np.multiply(baseline_estimate, 0.2) + np.multiply(svd_estimate, 0.8)
	rmse, mae = compute_error(actual_ratings, hybrid_estimate)

	print("\n" + "-" *5 + " Hybrid algorithm " + "-" *5)
	print("RMSE: ", rmse)
	print("MAE: ", mae)

	hybrid_estimate = hybrid_estimate.tolist()
	predictions = []
	for p,h in zip(prediction_baseline, hybrid_estimate):
		predictions.append((p[0], p[1], p[2], h, p[4]))

	return predictions

if __name__ == "__main__":

	df_train, df_test = data.get_train_test_data(new_sample = False)
	trainset, testset = convert_traintest_dataframe_forsurprise(df_train, df_test)

	# predictions = baseline(trainset, testset)
	# predictions = svdalgorithm(trainset, testset)
	# predictions = slopeOne(trainset, testset)
	# predictions = coClustering(trainset, testset)
	# predictions = kNNBasic(trainset, testset)
	# predictions = svdpp(trainset, testset)

	predictions = hybrid(trainset, testset)

	algorithm_recommendations,testdata_recommendations = movie_recommendation(predictions, n=10)
	# Print the recommended movies for each user
	#for user_id, user_ratings in algorithm_recommendations.items():
	#  print(user_id, [movie_id for (movie_id, estimated_rating) in user_ratings])
	[precision, recall, F_score] = precision_recall_calculation(predictions, threshold=3.5)
	print("Precision=", precision)
	print("Recall=", recall)
	print("F-Score=",F_score)