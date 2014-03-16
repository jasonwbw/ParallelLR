#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Test plr.models.logistic_regression.py
"""

import unittest
from abc import ABCMeta, abstractmethod
from plr.models.logistic_regression import LogisticRegression
from numpy import array

class TestLogisticRegression(unittest.TestCase):
	
	"""
	An abstract class for LogisticRegression's implement class' test
	"""

	__metaclass__ = ABCMeta

	def setUp(self):
		self.trainX = array([
			[1, 2, 3, 4], 
			[1, 3, 3, 4],
			[3, 4, 1, 2],
			[5, 3, 2, 1]])
		self.trainY = array(
			[1, 1, -1, -1])
		self.build_model(len(self.trainX[0]), 20, None)


	@abstractmethod
	def build_model(self, x_size, max_step, min_dt_abs):
		"""
		Build the lr_model like 'self.lr_model = ...'
		"""
		pass


	def tearDown(self):
		self.lr_model = None


	def testTrainMethod0(self):
		self.lr_model.train_method = 0
		pass


	def testPredictMethod0(self):
		self.lr_model.train_method = 0
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(True, self.predict(array([1, 1, 3, 4])) > 0)


	def testClassifyMethod0(self):
		self.lr_model.train_method = 0
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(1, self.classify(array([1, 1, 3, 4])))


	def testTrainMethod1(self):
		self.lr_model.train_method = 1
		pass


	def testPredictMethod1(self):
		self.lr_model.train_method = 1
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(True, self.predict(array([1, 1, 3, 4])) > 0)


	def testClassifyMethod1(self):
		self.lr_model.train_method = 1
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(1, self.classify(array([1, 1, 3, 4])))


	def testTrainMethod2(self):
		self.lr_model.train_method = 2
		pass


	def testPredictMethod2(self):
		self.lr_model.train_method = 2
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(True, self.predict(array([1, 1, 3, 4])) > 0)


	def testClassifyMethod2(self):
		self.lr_model.train_method = 2
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(1, self.classify(array([1, 1, 3, 4])))
#endclass TestLogisticRegression