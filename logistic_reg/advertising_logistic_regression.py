import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

ad = pd.read_csv('advertising.csv')
ad.head()
ad.info()
ad.describe()


# Exploratory Data Analysis
sns.set_style('whitegrid')
plt.hist(x='Age',data=ad,bins=30)

sns.histplot(x='Age',data=ad,bins=30)

ad['Age'].plot.hist(bins=30)


# *Jointplot showing Area Income versus Age.**
sns.jointplot(x='Age',y='Area Income',data=ad)


# Jointplot showing the kde distributions of Daily Time spent on site vs. Age.**
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad,kind='kde',color='red')


# Jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad,color='green')


# Pairplot with the hue defined by the 'Clicked on Ad' column feature.**
sns.pairplot(ad,hue='Clicked on Ad',palette='RdBu',diag_kind='hist')


# Logistic Regression

sns.heatmap(ad.isnull(),yticklabels=False,cmap='viridis',cbar=False)

ad.head()
ad.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)
ad.head()
ad.columns

X = ad.drop('Clicked on Ad',axis=1)
y = ad['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train and fit a logistic regression model on the training set.

log_model = LogisticRegression()
log_model.fit(X_train,y_train)

# Now predict values for the testing data.
predictions = log_model.predict(X_test)

# Classification report for the model.
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))