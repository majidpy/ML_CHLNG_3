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
ENCODE_FEATS = ('countrycode', 'browserid', 'devid')

###############     Functions definitions     ###############
def load_training_data(split_frac=0.1, drop_na=True, 
                       testing_mode=False, impute=True):
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

        1 X_test (numpy matrix): features of the model for testing training accuracy
        
        2 y_train (numpy matrix): the target value for train set
                
        3 y_test (numpy matrix): target values for testing 
        
        4 data_features(string list): main features names
        
        5 Encoders(string list): list of encoders for training set
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
        data_frame = data_frame.dropna(how='any')

    # Report on unique values in each feature
    num_unique_val_fetures(data_frame)
    
    # Cleaning up the date and time data
    parse_date_time_data(data_frame)
    
    # Getting rid of redundancy in browserid
    remove_browserid_redundancy(data_frame)
    
    # Label encoding the stringg/object data
    encoders_table = encode_train_labels(data_frame)
    
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
        
    # imputating data
    if impute:
        imp = preprocessing.Imputer()
        
        imp = imp.fit(X_train)
        X_train = imp.transform(X_train)
        
        if (split_frac > 0):
            imp = imp.fit(X_test)
            X_test = imp.transform(X_test)
        
    return (X_train, X_test, y_train, y_test, data_features, encoders_table)

def load_test_data(encoders_list, drop_na=True, 
                   testing_mode=False, impute=True):
    """
    Loads the test data from files and turns them into tables ready for 
    learning algorithms
    Args:
        encoders_list (list): list of encoders
        
        rest is self explanotory
        
    Returns:
        X_test (numpy matrix): features of the model (original and engineered) prepared 
        for feeding to learning algorithm
    """

    # Loading the dataset
    print("\nStarted loading test data ...")
    if testing_mode:
        data_frame = pd.read_csv('data/test_toy.csv')
    else:
        data_frame = pd.read_csv('data/test.csv')
    print("Data loaded\n")    

    if drop_na:
        data_frame = data_frame.dropna(how='any')

    # Cleaning up the date and time data
    parse_date_time_data(data_frame)
    
    # Getting rid of redundancy in browserid
    remove_browserid_redundancy(data_frame)
    
    # Label encoding the stringg/object data
    encode_train_labels(data_frame)
    
    # Turning data_fram into numerical matrix
    X = np.c_[data_frame['siteid'].values, data_frame['offerid'].values, 
              data_frame['category'].values, data_frame['merchant'].values, 
              data_frame['countrycode_le'].values, 
              data_frame['browserid_le'].values, data_frame['devid_le'].values, 
              data_frame['day'].values, data_frame['hour'].values]
    
    ID = np.c_[data_frame['ID'].values]
    # imputating data
    if impute:
        imp = preprocessing.Imputer()
        imp = imp.fit(X)
        X = imp.transform(X)
    
    return X, ID

def save_test_results(ID, y, file_name):
    """
    Saves the results of main test to an csv file
    Args:
        ID(numpy matrix size n): the unique ID of the inputs
        
        y(numpy matrix size n): the predicted values
        
    Returns:
        None
        
    Side effect:
        Saves file called "data/results.csv"
    """
    data_frame = pd.DataFrame(y, index=ID.ravel(), columns=['click'])
    data_frame.index.name = 'ID'
    data_frame.to_csv('data/' + file_name)

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
    

def encode_train_labels(df, has_nan=True):
    """
    Encodes the string data in TRAINING set with numerical labels
    
    Args:
        df(pandas dataframe): the input data
        
        rest is self-explanatory
        
    Returns:
        encoders_table (dictionary): contains lebeler, to be used with test data
        
    Side effect:
        changes the input df
    """    
    encoders_table = {} # Creating an empty dictionary
    
    for feat in ENCODE_FEATS:
        encoder = preprocessing.LabelEncoder()
        encoders_table[feat] = encoder
        if has_nan: # temporarily replacing NaN values
            series_nan_mask = df[feat].isnull()
            df[feat] = df[feat].fillna('temp_NaN')
            
        encoder.fit(df[feat])
        df[feat + '_le'] = encoder.transform(df[feat])
        
        if has_nan: # replacing back NaN values
            df[feat + '_le'] = df[feat + '_le'].where(series_nan_mask==False, np.nan)
        
        del df[feat]
        
    return encoders_table

def encode_test_labels(df, encoders_table, has_nan=True):
    """
    Encodes the string data in TEST set with numerical labels
    
    Args:
        df(pandas dataframe): the input data
        
        encoders_table(dictionary): containt the encoders used for encoding 
        training data
        
        rest is self-explanatory
        
    Returns:
        None
        
    Side effect:
        changes the input df
    """    

    for feat in ENCODE_FEATS:
        encoder = encoders_table[feat]
        if has_nan: # temporarily replacing NaN values
            series_nan_mask = df[feat].isnull()
            df[feat] = df[feat].fillna('temp_NaN')

        df[feat + '_le'] = encoder.transform(df[feat])
        
        if has_nan: # replacing back NaN values
            df[feat + '_le'] = df[feat + '_le'].where(series_nan_mask==False, np.nan)
        
        del df[feat] 
        
    
    
###############     ad-hoc Testing     ###############
if __name__ == "__main__":
    load_training_data(testing_mode=DEBUG_MODE, drop_na=False)



