from sklearn import datasets
import numpy as np
import multiprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
import time

iris = datasets.load_iris()
X = iris["data"] 
y = iris["target"]

param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': [1e-5, 1e-4, 5e-4, 1e-3, 2.3e-3, 5e-3, 1e-2, 1, 5, 10, 15, 20, 100]  
    }
]


gs = GridSearchCV(estimator = LogisticRegression(), param_grid = param_grid, scoring ='accuracy', cv = 10)
gs.fit(X, y)
print(gs.best_estimator_)

gs_scores = cross_val_score(gs.best_estimator_, X, y, scoring='accuracy', cv=10)
print('Best estimator CV average score: %.3f' % gs_scores.mean())