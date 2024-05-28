import unittest
import pandas as pd
from scripts.preprocess import load_and_preprocess_data

class TestPreprocess(unittest.TestCase):
    def test_load_and_preprocess_data(self):
        data = load_and_preprocess_data('path_to_test_file.csv')
        self.assertFalse(data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
