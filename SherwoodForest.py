# some of this code is from Jermey Howard's notebook.
# https://www.kaggle.com/code/jhoward/how-random-forests-really-work/
import numpy as np
import pandas as pd
import random

# this is  at static method which exists outside of the class definition
def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(df.mode().iloc[0], inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)
    return df

def _side_score(side, y):
    tot = side.sum()
    if tot <= 1:
        return 0
    return y[side].std() * tot

def min_col(df, column, response):
    col, y = df.iloc[:, column], response
    unq = col.dropna().unique()
    scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
    idx = scores.argmin()
    return unq[idx], scores[idx]

def score(col, y, split):
    lhs = col <= split
    return (_side_score(side=lhs, y=y) + _side_score(side=~lhs, y=y)) / len(y)

class SherwoodForest:
    def __init__(self, filepath, num_trees=10, num_rows=500):
        self.filepath = filepath
        self.train_data = None
        self.test_data = None
        self.y = None
        self.num_trees = num_trees
        self.num_rows = num_rows
        self.tree = []
        self.column_subsets = []

    def load_data(self, train_file, test_file, response):
        self.train_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + train_file)
        self.test_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + test_file)
        self.response = self.train_data[response]
        self.train_data = self.train_data.drop(response, axis=1)

    def process_data(self):
        self.train_data = proc_data(self.train_data)
        self.test_data = proc_data(self.test_data)

    def create_starting_nodes(self):
        # need to just separate the response completely
        self.column_subsets = [
            [num for num in set([
                random.randint(0, len(self.train_data.columns)-1) for _ in range(random.randint(2,len(self.train_data.columns)))
            ])
             ] for _ in range(self.num_trees)
        ]

    def create_tree(self):
        for cols in self.column_subsets:
            self.tree.append({c : min_col(df=self.train_data, column=c, response=self.response) for c in cols})

if __name__ == "__main__" :
    # intialize
    sf = SherwoodForest(filepath="/home/gigan/Python_Projects/Kaggle_Projs/titanic")
    # load data
    sf.load_data("train.csv","test.csv", 'Survived')
    # process data
    sf.process_data()
    # create node
    sf.create_starting_nodes()
    # create base of tree
    sf.create_tree()
