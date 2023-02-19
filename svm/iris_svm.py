import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
sns.pairplot(iris,hue='species',diag_kind='hist')

sns.kdeplot(x='sepal_width',y='sepal_length',data=iris[iris['species']=='setosa'],cmap='plasma',fill=True)

from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix

pred = model.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3,refit=True)

grid.fit(X_train,y_train)
grid.best_params_

grid_pred = grid.predict(X_test)

print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))