"""
Code for participating in HackerEarth Challenge 3
@author: Mojtaba
Method: Logistic Regression
"""
###############     Libraries     ###############
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

###############     Constants     ###############
DEBUG_MODE = True

###############     Functions definitions     ###############
def load_training_data(split_frac=0.1, testing_mode = False):
    
    # Loading the dataset
    print("Started loading training data ...")
    if testing_mode:
        data_frame = pd.read_csv('data/train_toy.csv')
    else:
        data_frame = pd.read_csv('data/train.csv')
    print("Data loaded")
    
    # Dropping data records with NaN
    # Can be improved later, but for now dropping them for simplicity
    data_frame = data_frame.dropna(how='any')
    
    # Finding the columns of the Dataframe
    data_features = data_frame.columns   
    
    
    return data_features
    
###############     ad-hoc Testing     ###############
f = load_training_data(testing_mode = DEBUG_MODE)
print(f)