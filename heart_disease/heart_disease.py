import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart_2020_cleaned.csv')
df.head()
df.drop(['MentalHealth','DiffWalking','Race','Asthma'],axis=1,inplace=True)
df

df1 = pd.get_dummies(df[['Smoking','AlcoholDrinking','Stroke','Sex','AgeCategory','Diabetic','PhysicalActivity','GenHealth','KidneyDisease','SkinCancer']],drop_first=True)
df1

data = pd.concat([df,df1],axis=1)
data.head()
data.drop(['Smoking','AlcoholDrinking','Stroke','Sex','AgeCategory','Diabetic','PhysicalActivity','GenHealth','KidneyDisease','SkinCancer'],axis=1,inplace=True)
data.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = data.drop('HeartDisease',axis=1)
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=50000)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
print('\n')
print(confusion_matrix(y_test,rfc_pred))