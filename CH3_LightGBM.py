"""
A script for training of HackerEarth CH3 data
"""
from CH3_data_loader import load_training_data, load_test_data, save_test_results
import lightgbm as lgb

###############     Constants     ###############
DEBUG_MODE = False 
DROP_NAN = False
SPLIT_FRACTION = 0.2
RUN_MAIN_TEST = True
REDUCED_MODEL = True
SAVE_FILE_NAME = 'results_light_gbm.csv'

###############     Training     ###############
data = load_training_data(split_frac=SPLIT_FRACTION, 
                        drop_na=DROP_NAN, testing_mode=DEBUG_MODE,
                        reduced=REDUCED_MODEL)

X_train = data[0]
X_test  = data[1] 
y_train = data[2] 
y_test  = data[3]
training_encoder = data[4]

print('Started training ...\n')
dtrain = lgb.Dataset(X_train, y_train)
dval = lgb.Dataset(X_test, y_test, reference=dtrain)
params = {
    'num_leaves' : 50,
    'learning_rate':0.01,
    'metric':'auc',
    'application':'binary',
    'early_stopping_round': 40,
    'max_depth':100,
    'num_threads':4,
    'verbose' : 1  
}
num_rounds = 200
lgb_model = lgb.train(params, dtrain, num_rounds, valid_sets=dval)
print('Training finished.\n')

###############     Testing main data     ###############    
if RUN_MAIN_TEST:
    X, ID = load_test_data(training_encoder, 
                           drop_na=False, testing_mode=DEBUG_MODE,
                           reduced=REDUCED_MODEL)
    print('Started running the main test ...')
    y = lgb_model.predict(X)
    print('Finished running the main test.')
    save_test_results(ID, y, SAVE_FILE_NAME)
    print('Saved results to \"data\\%s\"' %SAVE_FILE_NAME)