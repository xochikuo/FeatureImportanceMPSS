import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import ipdb
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sksurv.metrics import concordance_index_censored
from pyHSICLasso import HSICLasso

from numpy.random import seed


def load_data(d):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Use images for 3 and 8 only
    x_train = x_train[(y_train == 3) | (y_train == 8)]
    y_train = y_train[(y_train == 3) | (y_train == 8)]
    x_test = x_test[(y_test == 3 ) | (y_test == 8)]
    y_test = y_test[(y_test == 3 ) | (y_test == 8)]
    # Reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    #Relabel
    relabel = {8:1,
          3:0}
    y_train = np.array([relabel[xi] for xi in y_train])
    y_test = np.array([relabel[xi] for xi in y_test])    
    return x_train, y_train, x_test, y_test
    

class classification_model:
    def __init__(self):
        input_shape = (28, 28, 1)
        self.model = Sequential()
        self.model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1,activation=tf.nn.sigmoid))
        self.model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    def fit(self, x, y, *args, **kwargs):
        seed(1)
        self.model.fit(x, y, *args, **kwargs)
        
    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
    
    def predict(self, x):
        return self.model.predict(x)


class feature_selection:
    def __init__(self, x_train, y_train, scores):
        self.x_train = x_train
        self.y_train = y_train
        self.scores = scores
        seed(1)

    def select(self, method, censor):
        if method == 'MPSS':
            self.x_train, self.y_train, self.scores = self.truncate()
            return self.surv()
        if censor:
            self.x_train, self.y_train, self.scores = self.truncate()
        if method == 'RandomForest':
            return self.rf()
        if method == 'MutualInformation':
            return self.mi()
        if method == 'Lasso':
            return self.lasso()
        if method == 'pyHsicLasso':
            return self.Hsic(censor)
    
    def truncate(self):
        # ipdb.set_trace()
        keep = np.where(self.y_train == 1)
        return self.x_train[keep], self.y_train[keep], np.array(self.scores)[keep]

    
    def surv(self):
        y_structured = [ (True,s) for s in self.scores]
        y_structured = np.array(y_structured, dtype=[('class', 'bool_'), ('score', 'single')])
        x_train_nonzero = pd.DataFrame(self.x_train)
        x_train_nonzero = x_train_nonzero.loc[:, (x_train_nonzero != 0).any(axis=0)]
        estimator = CoxPHSurvivalAnalysis(alpha=0.1, verbose=1)
        estimator.fit(x_train_nonzero, y_structured)
        prediction = estimator.predict(x_train_nonzero)
        concordance = concordance_index_censored([True for x in y_structured], self.scores, prediction)
        print(concordance)
        importance = pd.DataFrame(estimator.coef_, columns=['coeff'])
        importance['coeff_abs'] = [math.fabs(c) for c in importance['coeff']]
        importance = importance.reset_index()
        importance = importance.sort_values('coeff', ascending=False)
        return importance
    
    def rf(self):
        rf_model = RandomForestClassifier()
        rf_model.fit(self.x_train, self.y_train)
        importance = pd.DataFrame(rf_model.feature_importances_, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False).reset_index()
        return importance
        
    def mi(self):
        # ipdb.set_trace()
        mi = pd.DataFrame(mutual_info_classif(self.x_train, self.y_train), columns=['mi'])
        mi = mi.sort_values('mi', ascending=False).reset_index()
        return mi
        
    
    def lasso(self):
        param_alpha = {'C':[0.1, 0.5, 1, 2, 3,5]}
        grid = GridSearchCV(LogisticRegression(penalty='l1',
                                                   solver='saga',
                                                   tol=0.1), 
                    param_grid=param_alpha, cv=3)
        grid.fit(self.x_train, self.y_train)
        C = grid.best_params_['C']
        clf = LogisticRegression(penalty='l1', 
                                      solver='saga', 
                                      tol=0.1, 
                                      C = C
        )
        clf.fit(self.x_train, self.y_train)
        sparsity = np.mean(clf.coef_ == 0) * 100
        lasso_coef = pd.DataFrame(clf.coef_[0], columns = ['coef'])
        lasso_coef['abs_coef'] = lasso_coef['coef'].apply(lambda x: math.fabs(x))
        lasso_coef = lasso_coef.sort_values('abs_coef', ascending=False)
        lasso_coef = lasso_coef.reset_index()
        return lasso_coef
    
    def Hsic(self, censor):
        data_predict = pd.DataFrame(self.x_train)
        # ipdb.set_trace()
        # if censor:
        data_predict['class'] = self.y_train
        # else:
        #     data_predict['class'] = self.y_train['class']
        data_predict[['class'] + data_predict.columns.values.tolist()[:-1]].to_csv('features_for_hsic.csv', index=False)
        hsic_lasso = HSICLasso()
        hsic_lasso.input('features_for_hsic.csv')
        hsic_lasso.classification(self.x_train.shape[1], B=50)
        return pd.DataFrame(np.array(hsic_lasso.get_features()), columns=['index'])

    
    
def mask_x(n_feat, importance, x_train, x_test):
    keep_features = importance.iloc[:n_feat, :]
    features = [int(x) for x in importance['index'].values.tolist()]
    numbered_pixel = [0] * (x_train.shape[1])
    numbered_pixel = [1 if i in features else 0 for i in range(x_train.shape[1])]
    numbered_pixel = np.array(numbered_pixel)
    current_features_used = sum(numbered_pixel)
    print(current_features_used)
    x_train = np.multiply(x_train.reshape(x_train.shape[0], 28, 28), numbered_pixel)
    x_test = np.multiply(x_test.reshape(x_test.shape[0], 28, 28), numbered_pixel)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, x_test


def save_results(list_accuracies, method, data_name, trunc, it):
    res = pd.DataFrame(list_accuracies, columns = ['n_features', 'test_accuracy'])
    res['method'] = method
    res['data'] = data_name
    res['truncated'] = trunc
    res['iteration'] = it
    res.to_csv('accuracies/accuracies_%s_%s_%s_%s.csv'%(method, data_name, trunc, it), index=False)
    return res


