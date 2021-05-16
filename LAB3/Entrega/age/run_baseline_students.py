#!/usr/bin/env

import os
import pickle as pkl

from lib.tools import *

from nn_torch_functions import *
from svr_functions import *

import numpy as np
import random as rn
import torch
import sklearn

# Fix random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12456)
rn.seed(12345)
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_svr(train_files, devel_files, test_files, feature_set):
	
	# Load data and labels
	X_train, y_train, _ = load_data(train_files)
	X_devel, y_devel, devel_filenames = load_data(devel_files)
	X_test, _ , test_filenames = load_data(test_files)

	# Data Pre-processing - Assuming data hasn't been pre-processed yet. Remove line if done
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(X_train) #normalize using preprocessor from train set
	X_train = scaler.transform(X_train)
	X_devel = scaler.transform(X_devel)
	X_test = scaler.transform(X_test)

	mae_results = []
	models = []
	
	for i in range(1):
		# Define Model Parameters
		if i == 0:
			parms = {'kernel': "rbf",
					'C'	 : 1,
					'g'	 : 0.01,
					'd'	 : 2}
		elif i == 1:
			parms = {'kernel': "rbf",
					'C'	 : 1,
					'g'	 : 0.01,
					'd'	 : 3}
		else:
			parms = {'kernel': "rbf",
					'C'	 : 1,
					'g'	 : 0.05,
					'd'  : 1}

		# Train Model
		print ('Training model ' + str(i) + ' ' + str(parms) + '...')
		model = train_svr(X_train, y_train, parms)
		models.append(model)

		# Compute predictions and metrics for train and devel
		train_mae, _ = test_svr(X_train, y_train, model)
		print('train - MAE: ', train_mae)

		devel_mae, _ = test_svr(X_devel, y_devel, model)
		print('dev - MAE: ', devel_mae)

		#sums train_mae and devel_mae to see which model has the best results
		aux_sum = train_mae + devel_mae
		mae_results.append(aux_sum)



		#when it reaches the end of the loop, 
		#compare which model is best and save only that
		if i == 2:
			min_mae = min(mae_results)
			index_min_mae = mae_results.index(min_mae)
			model = models[index_min_mae]

			print("\nBest model: {}".format(index_min_mae))


			# Compute predictions for test data
			predictions_test = model.predict(X_test)
			# Save test predictions
			save_predictions(test_filenames, predictions_test, 'test_result_svr.csv')


			# Compute predictions for devel data
			predictions_devel = model.predict(X_devel)
			# Save devel predictions
			save_predictions(devel_filenames, predictions_devel,'dev_result_svr.csv')
		

	# Save Model - After we train a model we can save it for later use
	pkl.dump(model, open('svr_model.pkl','wb'))


def run_nn(train_files, devel_files, test_files, feature_set):
	
	# define training parameters:
	epochs 		  = 40
	learning_rate = 0.1
	l2_decay 	  = 0.001
	batch_size    = 128
	dropout 	  = 0.1

	# define and criterion:
	criterion = nn.MSELoss()

	# initialize dataset with the data files and label files
	dataset = AgeDataset(train_files, devel_files, test_files)

	# Get number of classes and number of features from dataset
	n_features  = dataset.X.shape[-1]

	# initialize the model
	model = FeedforwardNetwork(n_features, dropout)
	model = model.to(device)

	# get an optimizer
	optimizer = "sgd"
	optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
	optim_cls = optims[optimizer]
	optimizer = optim_cls(
		model.parameters(),
		lr=learning_rate,
		weight_decay=l2_decay)

	# train the model
	model, train_mean_losses, valid_maes = train(dataset, model, optimizer, criterion, batch_size, epochs)

	# evaluate on train set
	train_X, train_y = dataset.X, dataset.y
	train_mae = evaluate(model, train_X, train_y)
	print('\nFinal Train MAE: %.3f' % (train_mae))

	# evaluate on dev set
	dev_X, dev_y = dataset.dev_X, dataset.dev_y
	dev_mae = evaluate(model, dev_X, dev_y)

	print('Final dev MAE: %.3f' % (dev_mae))
	print("\nParameters: lr = {}, ld = {}, bs = {}, d = {}".format(learning_rate, l2_decay, batch_size, dropout))

	# get predictions for test and dev set
	test_X = dataset.test_X

	predictions_test = predict(model, test_X)
	predictions_test = predictions_test.detach().cpu().numpy().squeeze()

	# Save test predictions
	save_predictions(dataset.test_files, predictions_test, 'test_result_nn.csv')


	# Compute predictions for devel data
	predictions_devel= predict(model, dev_X)
	predictions_devel = predictions_devel.detach().cpu().numpy().squeeze()	
	# Save devel predictions
	save_predictions(dataset.dev_files, predictions_devel, 'dev_result_nn.csv')


	# save the model
	torch.save(model, 'nn_model.pth')

	# plot training history
	plot_training_history(epochs, [train_mean_losses], ylabel='Loss', name='training-loss')
	plot_training_history(epochs, [valid_maes], ylabel='MAE', name='validation-metrics_mae')

def main():
	feature_set = "egemaps" # name of the feature set.
	
	train_files = 'train_agender.csv'
	devel_files = 'devel_agender.csv'
	test_files = 'test_voxceleb_pt.csv'
	
	# Run SVM - PART 2
	#run_svr(train_files, devel_files, test_files, feature_set)

	# Run NN - PART 3
	run_nn(train_files, devel_files, test_files, feature_set)

if __name__ == "__main__":
	main()