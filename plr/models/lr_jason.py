#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a tool to solve common regression problem by logistic regression(hereinafter referred to as LR).

"""

from numpy import array, random, exp, empty, dot, float32 as REAL
from logistic_regression import LogisticRegression

class LR(LogisticRegression):

	"""
	Single thread lr model from jason
	"""

	def train(self, X, Y):
		if self.train_method != 0 and self.train_method != 1 and self.train_method != 2: 
			#raise MethodError()
			return

		for i in xrange(self.max_step):
			if self.train_method == 0: 
				dt = - self.gd_train(X, Y, i)
				if i != 0 :
					self.update(dt)
				else:
					self.update(dt, decay_alpha = (self.decay != 1))
			elif self.train_method == 1:
				self.sgd_train(X, Y, i)
			elif self.train_method == 2:
				self.lbfgs_train(X, Y, i)


	def update(self, dt, decay_alpha = False):
		if self.min_dt_abs is None or abs(dt) > self.min_dt_abs:
			if decay_alpha :
				self.alpha *= self.decay
			self.W += self.alpha * dt


	def _sigmoid(self, x):
		if x >= 0:
			return 1. / ( 1  + exp( -x ) )
		else:
			return exp( x ) / ( 1 + exp( x ) )


	def gd_train(self, X, Y, step):
		vec_sigmoid = lambda X : array([self._sigmoid(x) for x in X], dtype = REAL) 
		tmp = Y * (vec_sigmoid( Y * dot(X, self.W.T) ) - 1)
		gt = dot(X , tmp.T)
		return gt


	def sgd_train(self, X, Y, step):
		multiply = lambda a, b: array([(a[i] * b[i]) for i in xrange(len(a))], dtype=REAL)
		for i in xrange(len(X)):
			gt = Y[i] * (self.__sigmoid(Y[i] * dot(X[i], self.W.T)) - 1) * X[i]
			if i == 0 and step != 0:
				self.update(- gt, decay_alpha = (self.decay != 1))
			else:
				self.update(- gt)


	def lbfgs_train(self, X, Y, step):
		# TODO: complete
		pass


	def predict(self, x):
		return ( self._sigmoid(dot(x, self.W.T)) - 0.5 ) * 2


	def save(self, fname):
		pass


	def load(self, fname):
		pass
#endclass LogisticRegression