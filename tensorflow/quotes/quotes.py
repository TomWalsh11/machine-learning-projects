import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('quotes.csv')

df.columns
df.head()
df.info()
df.describe().transpose()

# Sort Quote Status into accepted and not accepted and drop other rows
df['Quote Status'].value_counts()
df['Quote Status'] = df['Quote Status'].replace(['thinking_about_it', 'pending'], 'not_accepted')
df['Quote Status'].value_counts()

df['Quote Status'] = df['Quote Status'].apply(lambda x: 1 if x == 'approved' else (0 if x == 'not_accepted' else 2))
df['Quote Status'].value_counts()

df = df[df['Quote Status'] != 2]
df['Quote Status'].value_counts()

# Remove unwanted columns
df = df[['Quote Status','Brand','Buyout','Consignment','Contract Type','Estimated MSRP','Product Type','Retail Price','Store Credit']]

df.info()

# Look at correlation
df.corr(numeric_only=True)
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')

# See if the buyout and consignment data are duplicated
sns.boxplot(x='Quote Status',y='Consignment',data=df)
sns.boxplot(x='Quote Status',y='Buyout',data=df)
sns.scatterplot(x='Buyout',y='Consignment',data=df)
sns.scatterplot(x='Buyout',y='Store Credit',data=df)
sns.scatterplot(x='Buyout',y='Retail Price',data=df)

# It looks like the buyout column encapsulates consignment and store credit which makes sense, so drop these columns
df = df.drop(['Consignment','Store Credit'],axis=1)

# It looks like this column is largely 0, so we can drop this column
df['Estimated MSRP'].value_counts()
df = df.drop('Estimated MSRP',axis=1)

len(df)

# Explore retail price
df['Retail Price'].value_counts()

df['Retail Price'].isnull().value_counts()
df['Retail Price'].isna().value_counts()

df['Retail Price']

# dtype is object
df['Retail Price'][df['Retail Price'].str.len() > 7]

# We can see there are some strange values here so replace these
df['Retail Price'] = df['Retail Price'].apply(lambda x: float(x) if len(x) < 7 else 0)
df['Retail Price'].nunique()
df['Retail Price'].unique()

sns.displot(x='Retail Price',data=df,hue='Quote Status')

df.info()

# Now deal with categorical variables
df['Product Type'].value_counts()
# There are too many unique product types, so we'll drop this column
df = df.drop('Product Type',axis=1)

df['Brand'].value_counts()
df = df.drop('Brand',axis=1)

df['Contract Type'].value_counts()
# Get dummy variables for contract type
df = pd.get_dummies(data=df,columns=['Contract Type'],drop_first=True)

# Check for any remaining object columns
df.select_dtypes(['object']).columns

# Now split the data
from sklearn.model_selection import train_test_split
X = df.drop('Quote Status',axis=1).values
y = df['Quote Status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalise the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

X_train.shape
model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=80,validation_data=(X_test,y_test))

# Plot the losses
model.history.history
losses = pd.DataFrame(model.history.history)
losses.plot()

# Add in early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=1000,validation_data=(X_test,y_test),callbacks=[early_stop])

# Plot losses
losses = pd.DataFrame(model.history.history)
losses.plot()

# Evaluate model performance
from sklearn.metrics import classification_report,confusion_matrix
predictions = (model.predict(X_test) > 0.5).astype("int32")
predictions

print(classification_report(y_test,predictions))
print(confusion_matrix(y_true=y_test,y_pred=predictions))

# Add in dropout layers
model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(4,activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(4,activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=1000,validation_data=(X_test,y_test),callbacks=[early_stop])

# Plot losses
losses = pd.DataFrame(model.history.history)
losses.plot()

# Evaluate model performance
predictions = (model.predict(X_test) > 0.5).astype("int32")
predictions

print(classification_report(y_test,predictions))
print(confusion_matrix(y_true=y_test,y_pred=predictions))
