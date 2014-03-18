#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Test plr.models.logistic_regression.py
"""

import unittest
from plr.models.lr_jason import LR
from test_logistic_regression import TestLogisticRegression


class TestLrJason(TestLogisticRegression, unittest.TestCase):

	def build_model(self, x_size, max_step, min_dt_abs):
		super(TestLrJason, self).build_model(x_size, max_step, min_dt_abs)
		self.lr_model = LR(x_size , max_step = max_step, min_dt_abs = min_dt_abs)
#endclass TestLrJason


if __name__ == '__main__':
	unittest.main()