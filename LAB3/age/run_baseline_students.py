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
	X_devel, y_devel, _ = load_data(devel_files)
	X_test, _ , test_filenames = load_data(test_files)

	# Data Pre-processing - Assuming data hasn't been pre-processed yet. Remove line if done
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(X_test)
	X_train = scaler.transform(X_train)
	X_devel = scaler.transform(X_devel)
	X_test = scaler.transform(X_test)
	nfeat = X_train.shape[1]
	
	# Define Model Parameters
	parms = {'kernel': "rbf",
			 'C'	 : 0.5,
			 'g'	 : 0.5/nfeat,
			 'd'	 : 3}

	# Train Model
	# Inspect the function train_svm at svm_functions.py
	print ('Training model...')
	model = train_svr(X_train, y_train, parms)
	#raise NotImplementedError

	# Compute predictions and metrics for train and devel
	train_mae, _ = test_svr(X_train, y_train, model)
	print('train - MAE: ', train_mae)

	devel_mae, _ = test_svr(X_devel, y_devel, model)
	print('dev - MAE: ', devel_mae)

	# Compute predictions for test data
	predictions_test = model.predict(X_test)

	# Save test predictions
	save_predictions(test_filenames, predictions_test, feature_set + '_test_svm_predictions.csv')

	# Save Model - After we train a model we can save it for later use
	pkl.dump(model, open('svr_model.pkl','wb'))


def run_nn(train_files, devel_files, test_files, feature_set):
	
	# define training parameters:
	epochs 		  = 40
	learning_rate = 0.001
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

	# get predictions for test and dev set
	test_X = dataset.test_X

	predictions_test = predict(model, test_X)
	predictions_test = predictions_test.detach().cpu().numpy().squeeze()

	# Save test predictions
	save_predictions(dataset.test_files, predictions_test, feature_set + '_test_nn_predictions.csv')

	# save the model
	torch.save(model, 'nn_model.pth')

	# plot training history
	plot_training_history(epochs, [train_mean_losses], ylabel='Loss', name='training-loss')
	plot_training_history(epochs, [valid_maes], ylabel='MAE', name='validation-metrics_mae')

def main():
	feature_set = "egemaps" # name of the feature set.
	
	#
	train_files = 'train_agender.csv'
	devel_files = 'devel_agender.csv'
	test_files = 'test_voxceleb_pt.csv'
	
	# Run SVM - PART 2
	run_svr(train_files, devel_files, test_files, feature_set)

	# Run NN - PART 3
	run_nn(train_files, devel_files, test_files, feature_set)

if __name__ == "__main__":
	main()