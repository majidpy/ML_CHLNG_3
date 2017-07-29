"""
A script for training of HackerEarth CH3 data
"""
from CH3_data_loader import load_training_data, load_test_data, save_test_results
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

###############     Constants     ###############
DEBUG_MODE = False
DROP_NAN = False
SPLIT_FRACTION = 0.0
RUN_MAIN_TEST = True
SAVE_FILE_NAME = 'results_rand_forest.csv'
NUM_TREES = 200
NUM_CPU_CORES = 4
MIN_SAMPLE_LEAF = 50

###############     Training     ###############
data = load_training_data(split_frac=SPLIT_FRACTION, 
                        drop_na=DROP_NAN, testing_mode=DEBUG_MODE)

X_train = data[0]
X_test  = data[1] 
y_train = data[2] 
y_test  = data[3]
training_encoder = data[4]

print('Started training ...\n')
rfc = RandomForestClassifier(n_estimators=NUM_TREES, 
                             n_jobs=NUM_CPU_CORES,criterion='entropy',
                             min_samples_leaf=MIN_SAMPLE_LEAF)
rfc_model = rfc.fit(X_train, y_train)
print('Training finished.\n')

if SPLIT_FRACTION > 0:
    print('Started running cross validation ...\n')
    # Getting accuracy 
    pred_score = rfc_model.score(X_test, y_test)
    
    # Getting ROC score (used in competition)
    y_test_pred = rfc_model.predict_proba(X_test)
    y_test_pred_click = y_test_pred[:,1]
    pred_roc_c = metrics.roc_auc_score(y_test, y_test_pred_click)
    
    print('Finished running cross validation.\n')
    print('Cross validation:\n Score %.4f \t ROC %.4f\n' %(pred_score*100, pred_roc_c))

###############     Testing main data     ###############    
if RUN_MAIN_TEST:
    X, ID = load_test_data(training_encoder,
                           drop_na=False, testing_mode=DEBUG_MODE)
    print('Started running the main test ...')
    y = rfc_model.predict_proba(X)
    print('Finished running the main test.')
    save_test_results(ID, y[:,1], SAVE_FILE_NAME)
    print('Saved results to \"data\\%s\"' %SAVE_FILE_NAME)
    
