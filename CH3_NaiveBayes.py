"""
A script for training of HackerEarth CH3 data
"""
from CH3_data_loader import load_training_data
from sklearn.naive_bayes import GaussianNB

DEBUG_MODE = True
DROP_NAN = True
SPLIT_FRACTION = 0.2

data = load_training_data(split_frac=SPLIT_FRACTION, 
                        drop_na=DROP_NAN, testing_mode=DEBUG_MODE)

X_train = data[0]
X_test  = data[1] 
y_train = data[2] 
y_test  = data[3]

gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)

if SPLIT_FRACTION > 0:
    pred_score = gnb_model.score(X_test, y_test)
    print('predicted score on the cross validation set is %.4f' %(pred_score*100))