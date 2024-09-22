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


def _side_score(side, y):
    tot = side.sum()
    if tot <= 1:
        return 0
    return y[side].std() * tot


class RandomForest:
    def __init__(self, filepath):
        self.filepath = filepath
        self.train_data = None
        self.test_data = None

    def load_data(self, train_file, test_file):
        self.train_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + train_file)
        self.test_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + test_file)

    def process_data(self):
        self.train_data = proc_data(self.train_data)
        self.test_data = proc_data(self.test_data)

    def score(self, col, y, split):
        lhs = col <= split
        return (_side_score(lhs, y) + _side_score(~lhs, y))/len(y)

    def min_col(self, df, nm):
        col, y = df[nm], df[dep]
        unq = col.dropna().unique()
        scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
        idx = scores.argmin()
        return unq[idx], scores[idx]

