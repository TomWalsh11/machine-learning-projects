import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')
loans.info()
loans.describe()
loans.head()


# Exploratory Data Analysis

sns.set_style('darkgrid')
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,bins=30,color='blue',label='Fully Paid')
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,bins=30,color='red',label='Not Fully Paid')

plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(10,6))

sns.countplot(x='purpose',data=loans,hue='not.fully.paid')

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

plt.figure(figsize=(11,7))
sns.lmplot(x='fico',y='int.rate',data=loans,col='not.fully.paid',hue='credit.policy',palette='Set1')

loans.info()
loans.head()

cat_feats = ['purpose']
type(cat_feats)

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))