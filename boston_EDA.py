#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: EDA on Boston dataset
@author: josephinemiller
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns

""" Information about the columns:
    
crim: Per capita crime rate by town
zn: Proportion of residential land zoned for lots over 25,000 sq. ft
indus: Proportion of non-retail business acres per town
chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
nox: Nitric oxide concentration (parts per 10 million)
rm: Average number of rooms per dwelling
age: Proportion of owner-occupied units built prior to 1940
dis: Weighted distances to five Boston employment centers
rad: Index of accessibility to radial highways
tax: Full-value property tax rate per $10,000
ptratio: Pupil-teacher ratio by town
black: where Bk is the proportion of [people of African American descent] by town
lstat: Percentage of lower status of the population
medv: Median value of owner-occupied homes in $1000s

"""
## Start by reading the file:
df = read_csv('/Users/josephinemiller/Desktop/boston.csv')
print(df.head())


#Check its shape
print('The shape of the dataset =', df.shape)

#Check null counts (there are none in this case)
print('The null count of each column of the dataset are: ')
print(df.isnull().sum())

#Look at the shape of our y. In this case, 
#it follows a normal distribution with some outliers
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.displot(df['medv'])
plt.show()

#Now, let's look at a correlation matrix to select columns for X
#Usually you can choose columns for X without this step, but I will use it regardless

correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation Matrix with Heat Map")
plt.show()

#Based on what we know from the correlation matrix,
#I will use lstat and rm as my X columns. These two metrics have 
#significantly higher (absolute value) correlation coefficents than the others
#thus I am only using these two.
df2 = df[['rm', 'lstat', 'medv']]
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X = X[['rm', 'lstat']]

#I will use Standard Scaler on my X columns, then summarize everything
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

np.set_printoptions(precision=3)
print(rescaledX[0:5,:])


#Create a new df for the standardized values
dataStandDf = pd.DataFrame(rescaledX, columns = ['rm','lstat'])
dataStandDf['medv'] = y.values

print(df2.head())

#Let's take a look at the descriptive statistics for both df2 and dataStandDf

description = df2.describe()
print(description) 

descriptionScaled = dataStandDf.describe()
print(descriptionScaled)

#Let's also plot the histogram for both to see the distribution of data

df2.hist()
plt.show()
dataStandDf.hist()
plt.show()

#Now, let's look at using a Normalizer on our X (and summarize it)!

scalerN = Normalizer().fit(X)
normalizedX = scalerN.transform(X)
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])

#Let's make a df for the normalized data!

dataNormal = pd.DataFrame(normalizedX, columns = ['rm','lstat'])
dataNormal['medv'] = y.values

#Just as we did with the other verisons of our data, let's show the descriptive stats
#And the histogram (for distribution)

descriptionNorm = dataNormal.describe()
print(descriptionNorm)

dataNormal.hist()
plt.show()

#Let's do a scatter matrix for just our selected data first

plt.figure()
scatter_matrix(df2)
plt.show()


#Lastly, let's do a scatter matrix for all the data!
plt.figure()
scatter_matrix(df)
plt.show()

