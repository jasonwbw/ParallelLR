#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Test plr.models.logistic_regression.py
"""

from abc import ABCMeta, abstractmethod
from plr.models.logistic_regression import LogisticRegression
from numpy import array

class TestLogisticRegression(object):
	
	"""
	An abstract class for LogisticRegression's implement class' test
	"""

	__metaclass__ = ABCMeta

	def setUp(self):
		self.trainX = array([
			[1, 2, 33, 14], 
			[1, 3, 13, 44],
			[30, 25, 1, 2],
			[26, 40, 2, 1]])
		self.trainY = array(
			[1, 1, -1, -1])
		self.testX_1 = array([1, 1, 43, 24])
		self.testY_1 = 1
		self.testX_2 = array([43, 24, 1, 1])
		self.testY_2 = -1
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


	def testPredictMethod0(self):
		self.lr_model.train_method = 0
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(True, self.testY_1 * self.lr_model.predict(self.testX_1) > 0)
		self.assertEqual(True, self.testY_2 * self.lr_model.predict(self.testX_2) > 0)


	def testClassifyMethod0(self):
		self.lr_model.train_method = 0
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(self.testY_1, self.lr_model.classify(self.testX_1))
		self.assertEqual(self.testY_2, self.lr_model.classify(self.testX_2))


	def testTrainMethod1(self):
		self.lr_model.train_method = 1


	def testPredictMethod1(self):
		self.lr_model.train_method = 1
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(True, self.testY_1 * self.lr_model.predict(self.testX_1) > 0)
		self.assertEqual(True, self.testY_2 * self.lr_model.predict(self.testX_2) > 0)


	def testClassifyMethod1(self):
		self.lr_model.train_method = 1
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(self.testY_1, self.lr_model.classify(self.testX_1))
		self.assertEqual(self.testY_2, self.lr_model.classify(self.testX_2))


	def testTrainMethod2(self):
		self.lr_model.train_method = 2


	def testPredictMethod2(self):
		self.lr_model.train_method = 2
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(True, self.testY_1 * self.lr_model.predict(self.testX_1) > 0)
		self.assertEqual(True, self.testY_2 * self.lr_model.predict(self.testX_2) > 0)


	def testClassifyMethod2(self):
		self.lr_model.train_method = 2
		self.lr_model.train(self.trainX, self.trainY)
		self.assertEqual(self.testY_1, self.lr_model.classify(self.testX_1))
		self.assertEqual(self.testY_2, self.lr_model.classify(self.testX_2))
#endclass TestLogisticRegression