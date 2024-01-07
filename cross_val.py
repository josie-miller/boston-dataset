#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 8 18:22:44 2023
Cross-validation lab
@author: josephinemiller
"""

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = read_csv('/Users/josephinemiller/Desktop/boston.csv')
data1 = df.drop(df.columns[0], axis=1)
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]

# Prepare models
models = []

model = LinearRegression()
for num_features in [5, 7, 9, 10, 11, 12]:
    rfe = RFE(model, n_features_to_select=num_features)
    X_rfe = rfe.fit_transform(X, y)  # Apply feature selection to the data
    models.append(('RFE_' + str(num_features), X_rfe, model))

results = []

for idx, (name, X_rfe, model) in enumerate(models):
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X_rfe, y, cv=kfold, scoring='r2')
    results.append(cv_results)
    msg = f"RFE_{num_features}: {cv_results.mean():.6f} ({cv_results.std():.6f})"
    results.append(msg)
    print(msg)

# Boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')

# Convert numeric labels to strings for the x-axis
numeric_labels = [str(num_features) for num_features in [5, 7, 9, 10, 11, 12]]

ax = fig.add_subplot(111)
ax.set_xticks(range(len(numeric_labels)))
ax.set_xticklabels(numeric_labels)

pyplot.boxplot(results[::2])
pyplot.xlabel('Number of Features')
pyplot.ylabel('R2 Score')
pyplot.show()

'''Number of Features vs. R2 Score:

The box plot figure represents the distribution of R2 scores for each model based on the number of selected features.
Each box represents a model with a specific number of selected features (5, 7, 9, 10, 11, and 12).
Mean R2 Score:

The mean R2 score for each model is represented by the hor. line inside each box.
As the number of selected features increases, there is a  trend of higher mean R2 scores. This suggests that including more features generally improves the model's predictive performance.
Variance of R2 Score:

The spread or variance of R2 scores is indicated by the height of each box. A taller box shows higher variance.
Models with fewer selected features (e.g., 5 or 7) tend to have higher variance in R2 scores, meaning  the model's performance is less consistent.
Models with more selected features (e.g., 10, 11, or 12) have  lower variance, indicating more stable performance.
Optimal Number of Features:

Based on the mean R2 scores, the model with 12 selected features (RFE_12) has the highest mean R2 score, suggesting it performs the best on average.
Models with 10 and 11 selected features (RFE_10 and RFE_11) also have high mean R2 scores and lower variance, making them strong values.
Trade-off between Complexity and Performance:

The choice of the optimal number of features involves a trade-off between model complexity (more features) and predictive performance.
While adding more features can improve mean R2 scores, it may also increase model complexity and overfitting, especially for models with high variance.
Model Selection Considerations:

The choice of the optimal number of features should consider the specific problem, available data, and the trade-off between model performance and complexity.
For some applications, a simpler model with slightly lower performance (e.g., RFE_10 or RFE_11) may be preferred to enhance model interpretability and reduce the risk of overfitting.

Output: 
    
    RFE_12: 0.614113 (0.153889)
    RFE_12: 0.698587 (0.094982)
    RFE_12: 0.706612 (0.099858)
    RFE_12: 0.707160 (0.097988)
    RFE_12: 0.715264 (0.099194)
    RFE_12: 0.720546 (0.097957)
    
The algorithm comparison is linked.

'''
