import unittest
import numpy as np
from extreme.estimators import TailIndexEstimator, ExtremeQuantileEstimator

class TestEstimator(unittest.TestCase):
    def test_hill(self):
        X1 = np.array([1, 5, 7, 10, 15])
        estimator1 = TailIndexEstimator(X1)
        self.assertEqual(estimator1.hill(k_anchor=4), 2.1414958388964176)
        self.assertEqual(estimator1.hill(k_anchor=3), 0.7094105686164228)
        self.assertEqual(estimator1.hill(k_anchor=2), 0.5594074979928148)
        self.assertEqual(estimator1.hill(k_anchor=1), 0.40546510810816416)

    def test_longest_run(self):
        estimator = ExtremeQuantileEstimator(X=np.array([1, 5, 7, 10, 15]), alpha=0.1)
        x1 = np.array([1.221, 4.444, 3.555])
        x2 = np.array([1.221, 4.444, 3.55])
        x3 = np.array([1.221, 4.444, 3.555, np.inf])
        x4 = np.array([1.221, 4.444, 3.55, np.nan])

        self.assertEqual(estimator.longest_run(x1, 4), (1, 3))
        self.assertEqual(estimator.longest_run(x2, 4), (1, 3))
        self.assertEqual(estimator.longest_run(x3, 4), (1, 3))
        self.assertEqual(estimator.longest_run(x4, 4), (1, 3))



if __name__ == "__main__":
    unittest.main()