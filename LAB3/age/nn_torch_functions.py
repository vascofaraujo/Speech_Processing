#!/usr/bin/env python

import argparse
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from lib.tools import *

# Fix random seeds for reproducibility
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgeDataset(Dataset):
	
	def __init__(self, train_files, devel_files, test_files):

		#Load data and labels
		X_train, y_train, _ = load_data(train_files)
		X_dev, y_dev, _ = load_data(devel_files)
		X_test, _ , x_filenames = load_data(test_files)

		# Data Pre-processing - Assuming data hasn't been pre-processed yet
		'''
		Data Pre-processing:
		- This is an important step when preparing data for a classifier.
		- Normalizing or transforming data can greatly help the classifier achieve better results and train faster.
		- Sklearn has a several preprocessing functions that can be used to this end:
		- https://scikit-learn.org/stable/modules/preprocessing.html

		- If you have not done the feature processing at Part 1 - now it is a good time to do it.
		'''
		scaler = sklearn.preprocessing.StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_dev = scaler.transform(X_dev)
		X_test = scaler.transform(X_test)

		# normalising age labels by 100
		y_train = y_train / 100.0
		y_dev = y_dev / 100.0

		self.X = torch.tensor(X_train, dtype=torch.float).to(device)
		self.y = torch.tensor(y_train, dtype=torch.float).to(device)

		self.dev_X = torch.tensor(X_dev, dtype=torch.float).to(device)
		self.dev_y = torch.tensor(y_dev, dtype=torch.float).to(device)

		self.test_X = torch.tensor(X_test, dtype=torch.float).to(device)
		self.test_files = x_filenames

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]



class FeedforwardNetwork(nn.Module):
	def __init__(
			self, n_features, dropout, **kwargs):
		'''
		This function initializes the network. It defines its architecture.
			- n_features (int): number of features
			- dropout (float): dropout probability
		'''
		super(FeedforwardNetwork, self).__init__()
		'''
		The following block contains one linear layer and one activation function.

		One Linear layer is generically defined as nn.Linear(input_size, output_size).
		The number of neurons in the layer corresponds to the ouput size. Increasing the
		number of neurons in a layer increases the capability of the network to model the
		desired function. However, a very high number of neurons may lead the network to
		overfit, especially in situations where the training set is small.

		The activation functions add nonlinearities to the network. Some examples are:
		nn.ReLU(), nn.Tanh(), nn.Softmax().

		Between the nn.Linear() and the activation function, it is usual to include
		nn.BatchNorm1d(hidden_size), and after the adctivation function, it is usual to
		include nn.Dropout(p) to regularize the network.
		'''

		torch.manual_seed(1234)
		self.lin1 = nn.Sequential(
			nn.Linear(n_features, 32),
			#nn.BatchNorm1d(32),
			nn.ReLU()
			#nn.Dropout(p=dropout)
			)


		torch.manual_seed(1234)
		self.lin_out = nn.Linear(32, 1)

	def forward(self, x, **kwargs):
		"""
		This function corresponds to the forward pass, which means
		that the input is being propagated through the network, layer
		by layer.
			- x (batch_size x n_features): a batch of training examples
		"""

		output = self.lin1(x)
		output = self.lin_out(output)

		return output



def train_batch(X, y, model, optimizer, criterion, **kwargs):
	"""
	X (n_examples x n_features)
	y (n_examples): gold labels
	model: a PyTorch defined model
	optimizer: optimizer used in gradient step
	criterion: loss function
	"""

	model.train()

	# zero the parameter gradients
	optimizer.zero_grad()

	# forward step
	outputs = model.forward(X)

	loss = criterion(outputs.squeeze(), y.squeeze())

	# propagate loss backward
	loss.backward()

	# updtate the weights
	optimizer.step()

	return loss


def predict(model, X):
	"""X (n_examples x n_features)"""
	model.eval()
	pred = model.forward(X)  # scores shape: (n_examples x n_classes)

	return 100*pred


def evaluate(model, X, y):
	"""
	X (n_examples x n_features)
	y (n_examples): labels
	"""
	model.eval()

	# make the predictions
	y_hat = predict(model, X)

	# convert to cpu
	y_hat = y_hat.detach().cpu()
	y = 100*y.detach().cpu()

	# compute evaluation metrics
	print(y,y_hat)
	print("\n")
	mae = mean_absolute_error(y, y_hat)
	
	return mae


def train(dataset, model, optimizer, criterion, batch_size, epochs):

	train_dataloader = DataLoader(
		dataset, batch_size=batch_size, shuffle=True)

	dev_X, dev_y = dataset.dev_X, dataset.dev_y

	epochs = torch.arange(1, epochs + 1)
	train_mean_losses = []
	valid_maes = []
	train_losses = []

	for ii in epochs:
		print('\nTraining epoch {}'.format(ii))
		for X_batch, y_batch in train_dataloader:

			loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
			train_losses.append(loss)

		mean_loss = torch.tensor(train_losses).mean().item()
		print('Training loss: %.4f' % (mean_loss))

		train_mean_losses.append(mean_loss)

		val_mae = evaluate(model, dev_X, dev_y)

		valid_maes.append(val_mae)

		print('Valid mae: %.4f\n' % (val_mae) )

	return model, train_mean_losses, valid_maes

