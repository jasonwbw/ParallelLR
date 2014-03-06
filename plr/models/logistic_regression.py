#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a tool to solve common regression problem by logistic regression(hereinafter referred to as LR).

"""

from abc import ABCMeta, abstractmethod

class LogisticRegression(object):

	"""
	An abstract class for easy to use LR model.
	"""

	__metaclass__ = ABCMeta

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