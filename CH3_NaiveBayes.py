"""
A script for training of HackerEarth CH3 data
"""
from CH3_data_loader import load_training_data, load_test_data
from sklearn.naive_bayes import GaussianNB

###############     Constants     ###############
DEBUG_MODE = False
DROP_NAN = False
SPLIT_FRACTION = 0.2
RUN_MAIN_TEST = True

###############     Training     ###############
data = load_training_data(split_frac=SPLIT_FRACTION, 
                        drop_na=DROP_NAN, testing_mode=DEBUG_MODE)

X_train = data[0]
X_test  = data[1] 
y_train = data[2] 
y_test  = data[3]

print("Started training ...\n")
gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)
print("Training finished.\n")

if SPLIT_FRACTION > 0:
    print("Started running cross validation ...\n")
    pred_score = gnb_model.score(X_test, y_test)
    print("Finished running cross validation.\n")
    print('predicted score on the cross validation set is %.4f' %(pred_score*100))

###############     Testing main data     ###############    
if RUN_MAIN_TEST:
    X = load_test_data(drop_na=False)
    print("Started running the main test ...")
    y = gnb_model.predict(X)
    print("Finished running the main test ...")
    