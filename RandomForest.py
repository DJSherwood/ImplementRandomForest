# most of this code is from Jermey Howard's notebook.
# https://www.kaggle.com/code/jhoward/how-random-forests-really-work/
import numpy as np
import pandas as pd

# this is  at static method which exists outside of the class definition
def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(df.mode().iloc[0], inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)
    return df

class RandomForest:
    def __init__(self, filepath):
        self.filepath = filepath
        self.train_data = None
        self.test_data = None
        self.y = None
        self.num_trees = 10
        self.num_rows = 500
        self.tree = None
        self.subset_idxs = list()

    def load_data(self, train_file, test_file, y_name):
        self.train_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + train_file)
        self.test_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + test_file)

    def process_data(self):
        self.train_data = proc_data(self.train_data)
        self.test_data = proc_data(self.test_data)

    def create_subset_idxs(self):
        for i in self.num_trees:
            self.subset_idxs= []

    def _side_score(self, side):
        tot = side.sum()
        if tot <= 1:
            return 0
        return y[side].std() * tot

    def score(self, col, split):
        lhs = col <= split
        return (_side_score(lhs, self.y) + _side_score(~lhs, self.y)) / len(self.y)

    def min_col(self, x_name):
        unq = self.train_data[x_name].dropna().unique()
        scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
        idx = scores.argmin()
        return unq[idx], scores[idx]

    def create_tree(self, cols):
        self.tree = {i : min_col(df=self.subset_indicies, x_name=i) for i in cols}

import random
import pandas as pd
df = (pd.read_csv('./train.csv'))
l = random.sample(range(0,len(df.columns)),random.randint(0,len(df.columns)))
m = [x for x in l if x != 1]
print(m)

n = [ y for y in [9, 3, 2, 5, 7, 8, 10]]


# Create a 3x4 nested list with random numbers between 1-10
nested = [[random.randint(1, 10) for _ in range(4)] for _ in range(3)]
# Example output: [[7, 3, 9, 2], [5, 1, 8, 4], [6, 10, 2, 7]]

# Create a nested list with varying sublist lengths (2-5 elements each)
nested = [[random.randint(1, 10) for _ in range(random.randint(2, 5))] for _ in range(3)]
# Example output: [[3, 7], [4, 8, 1, 9], [2, 5, 6]]

# Using random.choice from a specific set of values
values = ['a', 'b', 'c', 'd']
nested = [[random.choice(values) for _ in range(3)] for _ in range(2)]
# Example output: [['b', 'a', 'c'], ['d', 'b', 'a']]

# Using random.uniform for floating point numbers
nested = [[round(random.uniform(0, 1), 2) for _ in range(2)] for _ in range(3)]
# Example output: [[0.23, 0.89], [0.45, 0.12], [0.67, 0.34]]