"""
A script for training of HackerEarth CH3 data
"""
from CH3_data_loader import load_training_data, load_test_data, save_test_results
from sklearn import tree

###############     Constants     ###############
DEBUG_MODE = False 
DROP_NAN = False
SPLIT_FRACTION = 0.0
RUN_MAIN_TEST = True
SAVE_FILE_NAME = 'results_dec_tree.csv'

###############     Training     ###############
data = load_training_data(split_frac=SPLIT_FRACTION, 
                        drop_na=DROP_NAN, testing_mode=DEBUG_MODE)

X_train = data[0]
X_test  = data[1] 
y_train = data[2] 
y_test  = data[3]
training_encoder = data[4]

print('Started training ...\n')
dtc = tree.DecisionTreeClassifier()
dtc_model = dtc.fit(X_train, y_train)
print('Training finished.\n')

if SPLIT_FRACTION > 0:
    print('Started running cross validation ...\n')
    pred_score = dtc_model.score(X_test, y_test)
    print('Finished running cross validation.\n')
    print('predicted score on the cross validation set is %.4f' %(pred_score*100))

###############     Testing main data     ###############    
if RUN_MAIN_TEST:
    X, ID = load_test_data(training_encoder, 
                           drop_na=False, testing_mode=DEBUG_MODE)
    print('Started running the main test ...')
    y = dtc_model.predict_proba(X)
    print('Finished running the main test.')
    save_test_results(ID, y[:,1], SAVE_FILE_NAME)
    print('Saved results to \"data\\%s\"' %SAVE_FILE_NAME)
    
