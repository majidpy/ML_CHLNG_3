"""
Code for participating in HackerEarth Challenge 3
@author: Mojtaba
Method: Logistic Regression
"""
###############     Libraries     ###############
import numpy as np
import pandas as pd
import datetime as dt

###############     Constants     ###############
DEBUG_MODE = True

###############     Functions definitions     ###############
def load_training_data(split_frac=0.1, testing_mode = False):
    """
    Loads the training data from files and turns them into tables ready for 
    learning algorithms
    Args:
        split_frac(float): the fraction of training data that is kept for cross check
        testing_mode(bool): Load from smaller test set of the full one
        
    Returns:
        A tuple with the following values in order
        0 X_train (numpy matrix): features of the model (original and engineered) prepared 
        for feeding to learning algorithm
        
        1 y_train (numpy matrix): the target value for train set
        
        2 X_train (numpy matrix): features of the model for testing training accuracy
        
        3 y_train (numpy matrix): target values for testing 
        
        4 data_features(string list): main features names
    """
    # Loading the dataset
    print("\nStarted loading training data ...")
    if testing_mode:
        data_frame = pd.read_csv('data/train_toy.csv')
    else:
        data_frame = pd.read_csv('data/train.csv')
    print("Data loaded\n")
    
    # Dropping data records with NaN
    # Can be improved later, but for now dropping them for simplicity
    data_frame = data_frame.dropna(how='any')

    # Report on unique values in each feature
    num_unique_val_fetures(data_frame)
    
    # Cleaning up the date and time data
    parse_date_time_data(data_frame)
    
    # Finding the columns of the Dataframe
    data_features = data_frame.columns
    
    return (data_frame, data_features)

def num_unique_val_fetures(df):
    """
    Finds the number of unique values in each feature
    
    Args:
        df(pandas dataframe): the input data
        
    Returns:
        None
        
    Output:
        prints unique values for each feature
    """
    cols = df.columns
    for c in cols:
        n = len(df[c].unique())
        print('%s has %d uniqe values' %(c, n))
        
def parse_date_time_data(df):
    """
    Changes the given dates in the dataframe into week days
    
    Args:
        df(pandas dataframe): the input data
        
    Returns:
        None
        
    Side effect:
        It create two new columns in the dataframe: "weekday" and "hour"
        And removes the original datetime string column
    """
    
    # Extracting the original datetime string column
    date_time = df['datetime']
    
    weekdays_array = np.zeros(len(date_time), dtype='int')
    hour_array     = np.zeros(len(date_time), dtype='int')
    
    cnt = 0
    for d in date_time:
        date = dt.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
        weekdays_array[cnt] = int(date.weekday())
        hour_array[cnt]     = int(date.hour)
        cnt += 1
       
    # Adding new columns
    df['weekday'] = pd.Series(weekdays_array, 
      index=df.index) # weekdays, Monday:0 ... Friday:6
    df['hour'] = pd.Series(hour_array, 
      index=df.index) # hour during the day
    
    # removing the original datetime columns
    del df['datetime']
    
    
###############     ad-hoc Testing     ###############
if __name__ == "__main__":
    load_training_data(testing_mode = DEBUG_MODE)



