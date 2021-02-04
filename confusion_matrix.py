import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
iris["target"]
X = iris["data"][:, 3:]  
y = (iris["target"] == 2).astype(np.int)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, Y_train)


y_true = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
y_pred = [0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
confusion_matrix(y_true, y_pred)
print(confusion_matrix)

cm = confusion_matrix(y_true=Y_test, y_pred= log_reg.predict(X_test))
print(classification_report(Y_test, log_reg.predict(X_test)))