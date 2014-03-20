#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a tool to solve common regression problem by logistic regression(hereinafter referred to as LR).

"""

from abc import ABCMeta, abstractmethod
from numpy import random

class LogisticRegression(object):
	
	#TODO : here lr is just for two class problem
	
	"""
	An abstract class for easy to use LR model  for classification(classify method) or regression(predict method) that y = 1 or -1

	Attributes:
	    W : the weight vector
		alpha : learning rate
		decay : decay for learning rate that alpha = alpha * (1 - decay * step)
		max_step : max number of learning step
		min_dt_abs : min abs value for dt, if abs(dt) < min_dt_abs, end the learning
		train_method : 
		    0 - Gradient Descent
		    1 - Stochastic gradient descent
		    2 - L-BFGS
	"""

	__metaclass__ = ABCMeta

	def __init__(self, x_size, alpha = 0.5, decay = 1.0, max_step = 200, min_dt_abs = None, train_method = 2):
		"""
		Init weight vector and other attributes

		Args:
		    x_size : size of x vector, and same with W vector
		    alpha : see LogisticRegression.alpha
		    decay : see LogisticRegression.decay
		    max_step : see LogisticRegression.max_step
		    min_dt_abs : see LogisticRegression.min_dt_abs
		    train_method : see LogisticRegression.train_method
		"""
		self.W = random.rand(x_size) - 0.5
		self.alpha = alpha
		self.decay = decay
		self.max_step = max_step
		self.min_dt_abs = min_dt_abs
		self.train_method = train_method


	@abstractmethod
	def train(self, X, Y):
		"""
		Train this model

		Args:
		    X : the matrix that hold a x in one line
		    Y : a one-dimensional vector that hold the result y for give x in the same line in X

		"""
		pass


	@abstractmethod
	def predict(self, x):
		"""
		Predit the result of give x vector

		Args:
		    x : a one-dimensional vector that hold one x

		Returns:
		    the result predit by the trained LR model
		"""
		pass


	def classify(self, x):
		"""
		Classify the give x vector

		Args:
		    x : a one-dimensional vector that hold one x

		Returns:
		    1 or -1 class
		"""
		if self.predict(x) > 0:
			return 1
		return -1


	@abstractmethod
	def save(self, fname):
		"""
		Save the trained model in the give file.

		Args:
		    fname : the file's name which will hold the model's parameters

		"""
		pass


	@abstractmethod
	def load(self, fname):
		"""
		Load model from the give file.

		Args:
		    fname : the file's name which holds the model's parameters

		"""
		pass
#endclass LogisticRegression