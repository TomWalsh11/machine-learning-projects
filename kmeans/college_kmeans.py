import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

college = pd.read_csv('College_Data',index_col=0)
college.head()
college.info()
college.describe()

sns.set_style('whitegrid')
plt.figure(figsize=(7,7))
sns.scatterplot(data=college,x='Room.Board',y='Grad.Rate',hue='Private',palette='coolwarm')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

sns.lmplot(x='Room.Board',y='Grad.Rate',data=college,hue='Private',palette='coolwarm',fit_reg=False,height=6,aspect=1)

plt.figure(figsize=(7,7))
sns.scatterplot(data=college,x='Outstate',y='F.Undergrad',hue='Private',palette='coolwarm')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

sns.set_style('darkgrid')
g = sns.FacetGrid(data=college,height=5,aspect=2,xlim=(0,25000),ylim=(0,70))
g.map_dataframe(sns.histplot,x='Outstate',hue='Private',palette='coolwarm',bins=20)

g = sns.FacetGrid(college,hue='Private',palette='coolwarm',height=6,aspect=2)
g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

sns.set_style('darkgrid')
g = sns.FacetGrid(data=college,height=5,aspect=2,xlim=(0,120),ylim=(0,70))
g.map_dataframe(sns.histplot,x='Grad.Rate',hue='Private',palette='coolwarm',bins=20)

college[college['Grad.Rate']>100]
college['Grad.Rate']['Cazenovia College'] = 100

sns.set_style('darkgrid')
g = sns.FacetGrid(data=college,height=5,aspect=2,xlim=(0,120),ylim=(0,70))
g.map_dataframe(sns.histplot,x='Grad.Rate',hue='Private',palette='coolwarm',bins=20)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(college.drop('Private',axis=1))

kmeans.cluster_centers_

ef converter(x):
    if x == 'Yes':
        return 1
    else:
        return 0

college['Cluster'] = college['Private'].apply(converter)
college.head()

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(college['Cluster'],kmeans.labels_))
print(classification_report(college['Cluster'],kmeans.labels_))