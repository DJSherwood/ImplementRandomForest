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
    df['Embarked'] = pd.Categorical(df.Embarked).codes
    df['Sex'] = pd.Categorical(df.Sex).codes
    df['Pclass'] = pd.Categorical(df.Pclass).codes
    return df

def _side_score(side, y):
    tot = side.sum()
    if tot <= 1:
        return 0
    return y[side].std() * tot

def score(x, y, split):
    lhs = x <= split
    return (_side_score(lhs, y) + _side_score(~lhs, y)) / len(y)

def min_col(df,predictor, response):
    x, y = df[predictor], df[response]
    unique_column = x.dropna().unique()
    scores = np.array([
        score(x, y, unique_value) for unique_value in unique_column if not np.isnan(unique_value)
    ])
    idx = scores.argmin()
    return unique_column[idx], scores[idx]

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
        self.predictors = []
        self.response = None

    def load_data(self, train_file, test_file, predictor_list, response):
        self.train_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + train_file)
        self.test_data = pd.read_csv(filepath_or_buffer=self.filepath + '/' + test_file)
        # if predictor list is empty, then use all the columns from the training se
        if not predictor_list:
            self.predictors = self.train_data.columns.to_list()
            self.predictors.remove('PassengerId')
        # otherwise, use the provided predictors
        else:
            self.predictors = predictor_list
        self.response = response

    def process_data(self):
        self.train_data = proc_data(self.train_data)
        self.test_data = proc_data(self.test_data)

    def create_starting_nodes(self):
        self.column_subsets = [
            random.sample(self.predictors, random.randint(1, len(self.predictors))) for _ in range(self.num_trees)
        ]

    def create_tree(self):
        for cols in self.column_subsets:
            self.tree.append({c : min_col(df=self.train_data.sample(frac=0.7), predictor=c, response=self.response) for c in cols})

if __name__ == "__main__" :
    # intialize
    sf = SherwoodForest(filepath="/home/gigan/Python_Projects/Kaggle_Projs/titanic")
    # load data
    sf.load_data("train.csv",
                 "test.csv",
                 ['Pclass','Sex','Age','Embarked','LogFare'],
                 'Survived')
    # process data
    sf.process_data()
    # create node
    sf.create_starting_nodes()
    # create base of tree
    sf.create_tree()
    print(sf.tree)
