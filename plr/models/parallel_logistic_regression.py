#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a mutil-thread tool to solve common regression problem by logistic regression(hereinafter referred to as LR).

"""

from abc import ABCMeta, abstractmethod
from logistic_regression import LogisticRegression

class MatrixSpliter(object):

	# TODO: comment

	def __init__(self, X, Y, m, n):
		# TODO: comment and all
		pass

	def get_submatrix(self, m, n):
		# TODO: comment and complete the matrix split, get the m line n column sub matrix
		pass


class CombineDt(object):

	# TODO: comment

	def __init__(self, m, n):
		# TODO: comment and all
		pass

	def add(m, n, Dt):
		# TODO: comment and add new result
		pass

	def combine():
		# TODO: comment and combine all results
		pass


class ParallelLogisticRegression(LogisticRegression):

	"""
	An abstract class for easy to use mutil-thread parallel LR model.
	"""

	__metaclass__ = ABCMeta

	def train(self, X, Y):
		# TODO: comment and complete the parallel training process
		pass


	@abstractmethod
	def node_compute(self, X, Y):
		"""
		The method called by children thread to compute Dt of give sub matrix

		Args:
		    X : the matrix that hold a x in one line
		    Y : a one-dimensional vector that hold the result y for give x in the same line in X

		Returns:
		    the Dt computed by the sub matrix
		"""
		pass

