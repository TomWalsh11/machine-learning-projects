# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the data
df = pd.read_csv(
    '/Users/thomaswalsh/Documents/Python/Data Science and Machine Learning/Data Science and Machine Learning Bootcamp/TensorFlow_FILES/DATA/lending_club_loan_two.csv')
df.info()

# Exploratory data analysis
sns.countplot(x='loan_status', data=df)

plt.figure(figsize=(12, 4))
sns.histplot(x='loan_amnt', data=df, alpha=0.2, bins=35)

# Correlation
df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap='viridis', annot=True)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='installment', y='loan_amnt', data=df)

sns.boxplot(x='loan_status', y='loan_amnt', data=df)

df['loan_amnt'].groupby(df['loan_status']).describe()

print(df['grade'].sort_values().unique())
print(df['sub_grade'].sort_values().unique())

sns.countplot(x='grade', data=df, hue='loan_status')

grade_order = df['sub_grade'].sort_values().unique()

plt.figure(figsize=(12, 4))
sns.countplot(x='sub_grade', data=df, palette='coolwarm', order=grade_order)

plt.figure(figsize=(12, 4))
sns.countplot(x='sub_grade', data=df, palette='coolwarm', order=grade_order, hue='loan_status')

plt.figure(figsize=(12, 4))
sns.countplot(x='sub_grade', data=df, palette='coolwarm', order=grade_order[25:], hue='loan_status')

df['loan_repaid'] = df['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0)
df[['loan_repaid', 'loan_status']].head(5)

df.corr()['loan_repaid'][:-1].sort_values().plot(kind='bar')

# Data preprocessing
df.head()
len(df)

df.isnull().sum()
df.isnull().sum() / len(df) * 100

len(df['emp_title'].unique())
df['emp_title'].value_counts()
df = df.drop('emp_title', axis=1)

emp_order = df['emp_length'].sort_values().unique()

plt.figure(figsize=(12, 6))
sns.countplot(x='emp_length', data=df, order=emp_order)

plt.figure(figsize=(12, 6))
sns.countplot(x='emp_length', data=df, order=df['emp_length'].value_counts().index, hue='loan_status')

len(df[df['loan_repaid'] == 1].groupby(['emp_length']))
df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()['loan_status'] / df['emp_length'].value_counts()
(df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()['loan_status'] / df[
    'emp_length'].value_counts()).plot(kind='bar')
df = df.drop('emp_length', axis=1)

df.head()
df.isnull().sum()

df[['title', 'purpose']]
df['purpose'].head(10)
df['title'].head(10)
df = df.drop('title', axis=1)

df.head()
df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()
df.groupby('total_acc')['mort_acc'].mean()

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
total_acc_avg[2.0]


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()

df.dropna(subset=['revol_util', 'pub_rec_bankruptcies'], inplace=True)
df.isnull().sum()

# Now deal with categorical variables
df.select_dtypes(include='object').columns

df['term'].value_counts()
df['term'] = df['term'].apply(lambda x: int(x[:3]))
df['term']

df = df.drop('grade', axis=1)
df.head()
df = pd.get_dummies(data=df, columns=['sub_grade'], drop_first=True)

df.head()
df.columns
df.select_dtypes(include='object').columns

df = pd.get_dummies(data=df, columns=['verification_status', 'application_type', 'initial_list_status', 'purpose'],
                    drop_first=True)
df.head()

df['home_ownership'].value_counts()
df['home_ownership'] = df['home_ownership'].apply(lambda x: x if x in ['MORTGAGE', 'RENT', 'OWN'] else 'OTHER')
df['home_ownership'].value_counts()
df = pd.get_dummies(data=df, columns=['home_ownership'], drop_first=True)

df['address']
df['zip_code'] = df['address'].apply(lambda x: x[-5:])
df['zip_code']
df['zip_code'].value_counts().unique()
df = pd.get_dummies(data=df, columns=['zip_code'], drop_first=True)
df = df.drop('address', axis=1)

df = df.drop('issue_d', axis=1)

df['earliest_cr_line']
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda year: int(year[-4:]))
df['earliest_cr_year']
df = df.drop('earliest_cr_line', axis=1)

df.select_dtypes(include='object').columns

# Train model
from sklearn.model_selection import train_test_split

df = df.drop('loan_status', axis=1)

X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

df = df.sample(frac=0.1, random_state=101)
print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X_train.shape

model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Dense(39, activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Dense(19, activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, batch_size=256, epochs=25, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save('tf_loan_model.h5')

# Evaluate performance
losses = pd.DataFrame(model.history.history)
losses
losses.plot()

predictions = (model.predict(X_test) > 0.5).astype('int32')

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Test on random customer
import random

random.seed(101)
random_ind = random.randint(0, len(df))

new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]
new_customer

model.predict(new_customer.values.reshape(1, 78) > 0.5).astype('int32')
df.iloc[random_ind]['loan_repaid']
