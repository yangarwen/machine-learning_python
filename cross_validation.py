from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] 
y = iris["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state = 46)

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state = 46)
softmax_reg.fit(X_train, Y_train)
pd.crosstab(Y_test, softmax_reg.predict(X_test),
            rownames=['label'], colnames=['predict'])

lr = LogisticRegression()
scores = cross_val_score(lr, X, y, scoring='accuracy', cv = 10)
print(scores, scores.mean())