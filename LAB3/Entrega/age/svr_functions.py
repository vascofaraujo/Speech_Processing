#!/usr/bin/env
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing


# Train Model
def train_svr(X, y, parms):
	'''
	Train Support Vector Machine:
	- In this function we define and train an SVM. 
	- A detailed guide of Sciki-Learn can be found in: https://scikit-learn.org/stable/
	
	In Scikit-Learn we have several functions to define an SVM, these include: (https://scikit-learn.org/stable/modules/svm.html#svm)
	 - SVR - SVR for classification, can be used with Linear, RBF and Polynomial Kernels, based on LibSVM
	 - LinearSVR - Faster implementation of the linear kernel, based on Liblinear
	 
	SVR requires (among others) the following parameters:
	 - Kernel: 'poly', 'rbf', 'sigmoid', 'linear', among others
	 - C: C works as a regularization parameter for the SVM.  
	      For larger values of C, a smaller margin will be accepted if the decision function is better at 
	      classifying all training points correctly. 
	      A lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. 
		 (Taken from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
	
	 - degree: Degree of the polynomial if the polynomial kernel is selected
	 - gamma: Kernel coefficient for RBF 
	
	Random State:
	 - If we are selecting the best parameters for our classifier we want to ensure all its intrinsic parameters are
	   initialized equally every time we train it with different parameters.
	'''
	
	if parms['kernel'] != 'linear':
		clf = SVR(kernel=parms['kernel'], C=parms['C'], degree=parms['d'], gamma=parms['g'], coef0=0.0, tol=0.001, epsilon=0.1, cache_size=200)
	else:
		clf = LinearSVR(C=parms['C'], epsilon=0.0, tol=0.0001, intercept_scaling=1.0, dual=True, verbose=0)
	
	clf.fit(X,y)

	return clf

# Compute Predictions and Metrics
def test_svr(X, y, clf):
	
	# After we have trained the model we can compute predictions on unseen data and use them to evaluate other metrics
	
	preds= clf.predict(X)
	
	# In this case we are using as metrics MAE.
	mae = mean_absolute_error(y, preds)

	return mae, preds
