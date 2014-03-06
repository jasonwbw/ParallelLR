#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a mutil-thread tool to solve common regression problem by logistic regression(hereinafter referred to as LR).

"""

from abc import ABCMeta, abstractmethod
from logistic_regression import LogisticRegression

class MatrixSpliter(object):

	"""
	Split the given matrix to m * n pieces

	Attributes:
	    X : the matrix that hold a x in one line
		Y : a one-dimensional vector that hold the result y for give x in the same line in X
		m : m pieces for line
		n : n pieces for column
	"""

	def __init__(self, X, Y, m, n):
		"""
		Init fuction, save the attributes

		Args:
		    X : see MatrixSpliter.X
		    Y : see MatrixSpliter.X
		    m : see MatrixSpliter.m
		    n : see MatrixSpliter.n
		"""
		pass

	def get_submatrix(self, m, n):
		"""
		Get the sub matrix in mth line and nth column

		Args:
		    m : mth pieces for line
		    n : nth pieces for column

		Returns:
		    (X, Y) tuple of sub matrix
		"""
		pass
#endclass MatrixSpliter


class CombineDt(object):

	"""
	Combine all dt computed from sub thread for sub matrix

	Attributes:
		m : m pieces for line
		n : n pieces for column
	"""

	def __init__(self, m, n):
		"""
		Init fuction, save the attributes

		Args:
		    m : see CombineDt.m
		    n : see CombineDt.n
		"""
		pass

	def add(m, n, Dt):
		"""
		Add new Dt result compute from the mth line and nth column sub matrix

		Args:
		    m : mth line
		    n : nth column
		    Dt : the Dt for given sub matrix
		"""
		pass

	def combine():
		"""
		Get the final dt to use for learning

		Returns:
		    final Dt vector whose dimensionality is same with x
		"""
		pass
#endclass CombineDt

class ParallelLogisticRegression(LogisticRegression):

	"""
	An abstract class for easy to use mutil-thread parallel LR model.
	"""

	__metaclass__ = ABCMeta

	def train(self, X, Y):
		# TODO: complete the parallel training process
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
#endclass ParallelLogisticRegression
