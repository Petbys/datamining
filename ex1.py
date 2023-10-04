'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
Student: Petter Byström
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('Exercise 1.a')
    print('------------')
    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":
    train = pd.read_csv('{}diabetes_train.csv'.format('/Users/petterbystrom/Documents/ETH/DataMining/Homework4/data/'))
    X_train = train.drop(columns=['type'])
    y_train= train['type']
    test = pd.read_csv('{}diabetes_test.csv'.format('/Users/petterbystrom/Documents/ETH/DataMining/Homework4/data/'))
    X_test = test.drop(columns=['type'])
    y_test= test['type']

   
    sc = StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)
    model=LogisticRegression(solver='lbfgs',max_iter=1000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    compute_metrics(y_test,y_pred)


    print('Exercise 1.b')
    print('------------')
    print('For the diabetes dataset I would choose Logistic regression. The LDA model gave more FP and less true Negatives than logistic regression meaning that the LDA model is more likely to give a positive diagnosis hence i would choose Logistic regression to not missdiagnos ')

    print('Exercise 1.c')
    print('------------')
    print('For another dataset i would choose Logistic regression. LDA is suited for datasets where the variables follow normal distribution. So if you dont want to make assumptions of a normally distributed dataset and be more flexible, Logistic regression is the method to choose')
    print('Exercise 1.d')
    print('------------')
    coefficients = model.coef_
    important_coeff= train.columns[np.argsort(abs(coefficients))[0][-1]],train.columns[np.argsort(abs(coefficients))[0][-2]]
    ods_npreg = math.exp(abs(coefficients)[0][0]/np.sqrt(sc.var_)[0])
    coef_npreg = abs(coefficients)[0][0]
    perc_npreg = (ods_npreg-1)*100
    print('The two attributes which appear to contribute the most to the prediction are {} and {}'.format(important_coeff[0],important_coeff[1]))
    print('The coefficient for npreg is {}. Calculating the exponential function results in {}, which amounts to an increase in diabetes risk of {} percent per additional pregnancy.'.format(round(coef_npreg,2),round(ods_npreg,2),round(perc_npreg,2)))