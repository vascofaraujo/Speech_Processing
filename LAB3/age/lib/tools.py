import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Load data from CSV file
def load_data(fp):

	data = pd.read_csv(fp)
	filenames = data['file']
	Y = data['age']
	X = data.drop(['age','gender','file','id'], axis=1)

	return X, Y, filenames


# Saves predictions
def save_predictions(filenames, predictions, output_path):
	pred_df = pd.DataFrame({'file_id': filenames, 'predictions': predictions})
	pred_df.to_csv(output_path, index=False)


# plot training history
def plot_training_history(epochs, plottable, ylabel='', name=''):
	plt.clf()
	plt.xlabel('Epoch')
	plt.ylabel(ylabel)
	if len(plottable) == 1:
		plt.plot(np.arange(epochs), plottable[0], label='Loss')
	elif len(plottable) == 2:
		plt.plot(np.arange(epochs), plottable[0], label='Acc')
		plt.plot(np.arange(epochs), plottable[1], label='UAR')
	else:
		raise ValueError('plottable passed to plot function has incorrect dim.')
	plt.legend()
	plt.savefig('%s.png' % (name), bbox_inches='tight')
