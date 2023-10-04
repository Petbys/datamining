#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
Student: Petter Byström
'''

import numpy as np
import sklearn.datasets
import math
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def split_data(X,y,attribute_index,theta):
    lower_X =[]
    upper_X = [] 
    lower_y =[]
    upper_y = [] 
    for i,row in enumerate(X):
        if row[attribute_index]<theta:
            lower_X.append(row)
            lower_y.append(y[i])
        else:
            upper_X.append(row)
            upper_y.append(y[i])
    return lower_X,upper_X,lower_y,upper_y

def compute_information_content(y):
    info= 0
    count = Counter(y)
    for i in count:
        info += count[i]/len(y)*math.log2(count[i]/len(y))
    return -info

def compute_information_a(X,y,attribute_index,theta):
    lower_X,upper_X,lower_y,upper_y=split_data(X,y,attribute_index,theta)
    return len(lower_X)/len(X)*compute_information_content(lower_y)+len(upper_X)/len(X)*compute_information_content(upper_y)

def compute_information_gain(X,y,attribute_index,theta):
    return compute_information_content(y)-compute_information_a(X,y,attribute_index,theta)



if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))
    
    ####################################################################
    # Your code goes here.
    # Gain(A) = info(D)-info_A(A)
    # info(D) = - sum(ylabel([1,2,3]|dataset))*log(2)P(ylabel([1,2,3]|dataset))

    ####################################################################
    
    print('Exercise 2.b')
    print('------------')
    splits = [[0,5.5],[1,3.0],[2,2.0],[3,1.0]]
    information_gain =[]
    for i in splits:
        information_gain.append(compute_information_gain(X,y,i[0],i[1]))
    information_gain = [round(i,2) for i in information_gain]
    
    print('Split ( sepal length (cm) < 5.5 ): information gain = {} \n Split ( sepal width (cm)  < 3.0 ): information gain = {} \n Split ( petal length (cm) < 2.0 ): information gain = {} \nSplit ( petal width (cm)  < 1.0 ): information gain = {}'.format(information_gain[0],information_gain[1],information_gain[2],information_gain[3]))

    print('Exercise 2.c')
    print('The attribute with the maximum gain is the best split. Hence i would choose petal length and width which both had and information gain of 0.92 ')
    print('------------')

    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################
    np.random.seed(42)
    accuracy=[]
    importance = []
    cv = KFold(n_splits=5,shuffle=True)
    for i, (train_index, val_index) in enumerate(cv.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = DecisionTreeClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy.append(accuracy_score(y_val,y_pred))
        importance.append(model.feature_importances_)
    mean_accuracy = np.mean(accuracy)



    print('Accuracy score using cross-validation')
    print('-------------------------------------\n')
    print(f'{round(mean_accuracy*100,2)}%')

    print('')
    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')
    print('The two most importance features are petal length and width')
    #reduced data set
    red_X = X[y!=2]
    red_y = y[y!=2]
    importance_red = []
    for i, (train_index, val_index) in enumerate(cv.split(red_X)):
        X_train, X_val = red_X[train_index], red_X[val_index]
        y_train, y_val = red_y[train_index], red_y[val_index]
        model = DecisionTreeClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy.append(accuracy_score(y_val,y_pred))
        importance_red.append(model.feature_importances_)
    mean_accuracy = np.mean(accuracy)

    print('')
    print('Feature importances for _reduced_ data set')
    print('------------------------------------------\n')
    print('only petal length is important. Meaning that the only petal length is used to distiguish between the species setosa and virginica')
