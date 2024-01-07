#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: MLR on boston dataset
@author: josephinemiller
"""


import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = read_csv('/Users/josephinemiller/Desktop/boston.csv')
print(df.head())
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation Matrix with Heat Map")
plt.show()

"""The inputs that have a correlation output where |x|>0.5 are 
'rm', 'pratio', and 'lstat'."""

df.hist()
plt.show()

"""" Based on what we see from the histogram, 
'rm', 'pratio', 'lstat', 'black', and 'nox' may be good X-values"""

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - df with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X, y)

print('resulting features:')
print(result)

'''the resulting features are:
['lstat', 'rm', 'ptratio', 'dis', 'nox', 'chas', 'black', 'zn']'''

''' Here we will relook at the visual linear correlation
with the best features 'lstat' and 'rm' from above'''

plt.figure(figsize=(20, 5))

features = ['lstat', 'rm']
target = df['medv']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    X1 = df[col]
    y1 = target
    plt.scatter(X1, y1, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

X1=df[features]
y1=target

''' #Post note, after many tries it is clear
that these two features together are the best for this model'''

X_train, X_test, Y_train, Y_test = train_test_split(X1, y1, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

y_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_predict)))
r2 = r2_score(Y_train, y_predict)

print("The regular model performance for training set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The regular model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

"Now let's do a version with the X values standard-scaled"

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

df_standard = pd.DataFrame(rescaledX)
df_standard['medv'] = y.values

X_train, X_test, y_train, y_test = train_test_split(rescaledX, y1, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)
# Evaluate the model

rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2 = r2_score(y_train, y_pred)

print("The scaled model performance for training set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
r2 = r2_score(y_test, y_test_pred)

print("The scaled model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

"RMSE is 4.94952722768856"
"R2 score is 0.6659408703343035"

"""Standardization increases performance metrics in machine learning by 
ensuring that all features have a comparable scale.
This is essential because many machine learning algorithms, including linear regression, 
are sensitive to the scale of the input features. Standardization helps prevent 
certain features from dominating the models learning 
process and enables more effective convergence, resulting in improved model performance and interpretability.
This is why our RMSE and R2 scores are better wiht the scaled data."""

"Now let's repeat with normalization!"

scalerN = Normalizer().fit(X)
normalizedX = scalerN.transform(X)
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])


df_normal = pd.DataFrame(normalizedX)
df_normal['medv'] = y.values

X_train, X_test, y_train, y_test = train_test_split(normalizedX, y1, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

"Now let's evaulate this model!"

rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2 = r2_score(y_train, y_pred)

print("The normalized model performance for training set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
r2 = r2_score(y_test, y_test_pred)

print("The nomalized model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
"RMSE is 4.7131798747898115"
"R2 score is 0.6970827962937747"

""" Normalization, like standardization, enhances machine learning 
performance by scaling features; however, it additionally constrains 
them to a common range, [0, 1]. In the Boston dataset, 
normalization might have led to higher R-squared (R2) and 
lower root mean square error (RMSE) compared to standard scaling 
and the unprocessed dataset because it's particularly effective 
when dealing with features of varying ranges. 
In the Boston dataset, certain features may have inherently 
different scales, and normalization helps ensure that they 
all contribute meaningfully to the regression model, 
resulting in improved predictive accuracy (higher R2) and 
reduced error (lower RMSE)."""

"""
Here is the full output:
runfile('/Users/josephinemiller/untitled0.py', wdir='/Users/josephinemiller')
   Unnamed: 0     crim    zn  indus  chas  ...  tax  ptratio   black  lstat  medv
0           1  0.00632  18.0   2.31     0  ...  296     15.3  396.90   4.98  24.0
1           2  0.02731   0.0   7.07     0  ...  242     17.8  396.90   9.14  21.6
2           3  0.02729   0.0   7.07     0  ...  242     17.8  392.83   4.03  34.7
3           4  0.03237   0.0   2.18     0  ...  222     18.7  394.63   2.94  33.4
4           5  0.06905   0.0   2.18     0  ...  222     18.7  396.90   5.33  36.2

[5 rows x 15 columns]
Add  lstat                          with p-value 5.0811e-88
/Users/josephinemiller/untitled0.py:66: FutureWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
  new_pval = pd.Series(index=excluded)
/Users/josephinemiller/untitled0.py:66: FutureWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
  new_pval = pd.Series(index=excluded)
Add  rm                             with p-value 3.47226e-27
Add  ptratio                        with p-value 1.64466e-14
Add  dis                            with p-value 1.66847e-05
Add  nox                            with p-value 5.48815e-08
Add  chas                           with p-value 0.000265473
Add  black                          with p-value 0.000771946
Add  zn                             with p-value 0.00465162

resulting features:
['lstat', 'rm', 'ptratio', 'dis', 'nox', 'chas', 'black', 'zn']

(404, 2)
(102, 2)
(404,)
(102,)

The regular model performance for training set
RMSE is 5.6371293350711955
R2 score is 0.6300745149331701


The regular model performance for testing set
RMSE is 5.137400784702911
R2 score is 0.6628996975186953

[[-1.729 -0.42   0.285 -1.288 -0.273 -0.144  0.414 -0.12   0.14  -0.983
  -0.667 -1.459  0.441 -1.076]
 [-1.722 -0.417 -0.488 -0.593 -0.273 -0.74   0.194  0.367  0.557 -0.868
  -0.987 -0.303  0.441 -0.492]
 [-1.715 -0.417 -0.488 -0.593 -0.273 -0.74   1.283 -0.266  0.557 -0.868
  -0.987 -0.303  0.396 -1.209]
 [-1.708 -0.417 -0.488 -1.307 -0.273 -0.835  1.016 -0.81   1.078 -0.753
  -1.106  0.113  0.416 -1.362]
 [-1.701 -0.412 -0.488 -1.307 -0.273 -0.835  1.229 -0.511  1.078 -0.753
  -1.106  0.113  0.441 -1.027]]

The scaled model performance for training set
RMSE is 4.639693462368617
R2 score is 0.7522054567340507


The scaled model performance for testing set
RMSE is 4.94952722768856
R2 score is 0.6659408703343035

[[2.000e-03 1.264e-05 3.600e-02 4.620e-03 0.000e+00 1.076e-03 1.315e-02
  1.304e-01 8.179e-03 2.000e-03 5.919e-01 3.060e-02 7.937e-01 9.959e-03]
 [4.237e-03 5.785e-05 0.000e+00 1.498e-02 0.000e+00 9.935e-04 1.360e-02
  1.671e-01 1.052e-02 4.237e-03 5.126e-01 3.771e-02 8.408e-01 1.936e-02]
 [6.439e-03 5.857e-05 0.000e+00 1.517e-02 0.000e+00 1.007e-03 1.542e-02
  1.311e-01 1.066e-02 4.293e-03 5.194e-01 3.820e-02 8.431e-01 8.649e-03]
 [8.779e-03 7.105e-05 0.000e+00 4.785e-03 0.000e+00 1.005e-03 1.536e-02
  1.005e-01 1.331e-02 6.584e-03 4.872e-01 4.104e-02 8.661e-01 6.453e-03]
 [1.090e-02 1.506e-04 0.000e+00 4.754e-03 0.000e+00 9.988e-04 1.559e-02
  1.182e-01 1.322e-02 6.543e-03 4.841e-01 4.078e-02 8.656e-01 1.162e-02]]

The normalized model performance for training set
RMSE is 4.619010732477142
R2 score is 0.7544097594269125


The nprmalized model performance for testing set
RMSE is 4.7131798747898115
R2 score is 0.6970827962937747

"""


