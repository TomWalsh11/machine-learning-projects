import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers')

customers.head()
customers.describe()
customers.info()


# Exploratory Data Analysis
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')

sns.pairplot(customers)

sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)

customers.head()
customers.columns

X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)

print('Coefficients:')
print(lm.coef_)

lm.coef_

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))

print('MAE: ',metrics.mean_absolute_error(y_test,predictions))
print('MSE: ',metrics.mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

metrics.explained_variance_score(y_test,predictions) # This is the R squared

# Residuals
sns.displot((y_test-predictions),bins=50,kde=True)

cdf = pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coefficients'])
cdf

# App generates better return, but website clearly needs work. Depends!