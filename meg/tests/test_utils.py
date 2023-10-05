import numpy as np
import unittest

import datasets.utils as utils


class TestUtils(unittest.TestCase):
    

    def test_mean_variance(self):
        data_iter = [
            np.random.randn(100).reshape(10,10)
            for _ in range(1000)
        ]
        
        data = np.stack(data_iter)
        expected_mean = data.mean()
        expected_var = data.var()

        mean, var = utils.mean_variance(data_iter)

        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(var, expected_var)
        
    def test_arbitrary_order_twice(self):
        data = np.arange(100)
        np.random.shuffle(data)
        
        a = utils.arbitrary_order(data)
        b = utils.arbitrary_order(a)

        self.assertListEqual(a,b)

    def test_arbitrary_order_different_seeds(self):
        data = np.arange(100)
        np.random.shuffle(data)
        
        a = utils.arbitrary_order(data, 42)
        b = utils.arbitrary_order(data, 43)

        same = True
        for x,y in zip(a,b):
            if x != y:
                same = False
                break
        
        self.assertFalse(same)






if __name__ == '__main__':
    unittest.main()

