"""
Code for participating in HackerEarth Challenge 3
@author: Mojtaba
Method: Logistic Regression
"""
###############     Libraries     ###############
import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import preprocessing

###############     Constants     ###############
DEBUG_MODE = True

###############     Functions definitions     ###############
def load_training_data(split_frac=0.1, drop_na=True, testing_mode=False):
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
    
    if drop_na:
        # Dropping data records with NaN
        # Can be improved later, but for now dropping them for simplicity
        data_frame = data_frame.dropna(how='any')

    # Report on unique values in each feature
    num_unique_val_fetures(data_frame)
    
    # Cleaning up the date and time data
    parse_date_time_data(data_frame)
    
    # Getting rid of redundancy in browserid
    remove_browserid_redundancy(data_frame)
    
    # Label encoding the stringg/object data
    encode_labels(data_frame)
    
    # Finding the columns of the Dataframe
    data_features = data_frame.columns
    
    # Turning data_fram into numerical matrix
    X = np.c_[data_frame['siteid'].values, data_frame['offerid'].values, 
              data_frame['category'].values, data_frame['merchant'].values, 
              data_frame['countrycode_le'].values, 
              data_frame['browserid_le'].values, data_frame['devid_le'].values, 
              data_frame['day'].values, data_frame['hour'].values]

    y = data_frame['click'].values
                  
    # Splitting dataset into train and test 
    if (split_frac > 0):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_frac)
    else:
        X_train = X
        X_test  = []
        y_train = y
        y_test  = []
        
    return (X_train, X_test, y_train, y_test, data_features)

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
    
    df['day']=pd.to_datetime(df['datetime']).apply(lambda x: x.dayofweek)
    df['hour']=pd.to_datetime(df['datetime']).apply(lambda x: x.hour)
       
    # Adding new columns
    df['weekday'] = pd.Series(weekdays_array, 
      index=df.index) # weekdays, Monday:0 ... Friday:6
    df['hour'] = pd.Series(hour_array, 
      index=df.index) # hour during the day
    
    # removing the original datetime columns
    del df['datetime']

def remove_browserid_redundancy(df):
    """
    Removes identical values for the browser id
    
    Args:
        df(pandas dataframe): the input data
        
    Returns:
        None
        
    Side effect:
        changes the input df
    """
    
    df['browserid'].replace(to_replace=['Mozilla Firefox', 'Mozilla'], 
      value='Firefox', inplace=True) # Firefox
    df['browserid'].replace(to_replace=['Google Chrome'], 
      value='Chrome', inplace=True) # Chrome
    df['browserid'].replace(to_replace=['InternetExplorer', 'Internet Explorer'], 
      value='IE', inplace=True) # IE
      
    print("\nBrowers narrowed down to ", df['browserid'].unique(), "\n")

def encode_labels(df):
    encoder = preprocessing.LabelEncoder()
    for feat in ('countrycode', 'browserid', 'devid'):
        encoder.fit(df[feat])
        df[feat + '_le'] = encoder.transform(df[feat])
        del df[feat]
    
###############     ad-hoc Testing     ###############
if __name__ == "__main__":
    load_training_data(testing_mode = DEBUG_MODE)



