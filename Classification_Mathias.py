# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:09:14 2024

@author: Mathi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:47:52 2024

@author: Mathi
"""

#%%
import numpy as np 
import statsmodels.api as sm 
import pylab as py 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#%%
df_white = pd.read_csv("winequality-white.csv", sep= ";")
df_white["Type"] = "White"
df_red= pd.read_csv("winequality-red.csv")
df_red["Type"] = "Red"
df_red_describtion = df_red.describe()
df_white_describtion = df_white.describe()

df = pd.concat([df_red,df_white])

describtion_red = df_red.describe()
describtion_white = df_white.describe()
#%%
#Variables

Depth = 25
Neighbours = 2
#EDA




#%%

fig, axes = plt.subplots(3, 4, figsize=(30, 20))  


axes = axes.flatten()
# Plot QQ plots
for i in range(10):
    sm.qqplot(df.iloc[:, i], line='45', ax=axes[i])
    axes[i].set_title(df.columns[i])


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
#%%



#Very clearly are none of the parameters normally distributed 
#We should check for homogeny in the covariance matrix
#Since our parameters are not normally distributed we shall use levenes test


first = df.iloc[:, 0].values
second = df.iloc[:, 1].values
third = df.iloc[:, 2].values
fourth = df.iloc[:, 3].values
fifth = df.iloc[:, 4].values
sixth = df.iloc[:, 5].values
seventh = df.iloc[:, 6].values
eighth = df.iloc[:, 7].values
ninth = df.iloc[:, 8].values
tenth = df.iloc[:, 9].values
eleventh = df.iloc[:, 10].values

# Performing Levene's test
res = stats.levene(first, second,third,fourth,fifth,sixth,seventh,eighth,ninth,tenth,eleventh)
print(res.pvalue)
#The pvalue is so small that the program rounds it to 0.0, so we will reject the null-hypthosis. 
#The parameters do not have homogeny in their covariance matrix for atleast one pair.
#%%

#%%



df['Grade'] = df['quality'].apply(lambda x: 'Not Great' if x <= 6 else 'Great')
df.to_excel("C:/Users\Mathi\Documents\Studie\2 semester\DM868, DM870, DS804 Data mining and machine learning\Projekt\Projekt 2", index=False)
#Turn the categories Type into values. 1 if White and 0 if Red
df['WineType'] = df['Type'].apply(lambda x: 1 if x == "White" else 0)
df = df.drop(columns=["Type"])
df = df.drop(columns=["quality"])
# Features and target variable
X = df.drop(columns=["Grade"])  # All columns except 'Grade'
y = df["Grade"]  # Target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%% RandomForestClassifier
#Train the model using winetype RandomForestClassifier

clf = RandomForestClassifier(max_depth=Depth, random_state=0, n_estimators = 500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
RFC_report = classification_report(y_test, y_pred)
print("Classification Report RFC:\n", RFC_report)
#%% Logistic regression
log_reg = LogisticRegression(random_state=0, max_iter=25000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
LR_report = classification_report(y_test, y_pred)
print("Classification Report LR:\n", LR_report)
#%% KNN
knn = KNeighborsClassifier(n_neighbors=Neighbours)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
KNN_report = classification_report(y_test, y_pred)
print("Classification Report KNN:\n", KNN_report)


#%% Decision Tree

decision_tree = DecisionTreeClassifier(max_depth=Depth, random_state=0)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
DT_report = classification_report(y_test, y_pred)
print("Classification Report DT:\n", DT_report)

#%% Now without using the WineType as an indicator
df = df.drop(columns=["WineType"])
X = df.drop(columns=["Grade"])  # All columns except 'Grade'
y = df["Grade"]  # Target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%% RandomForestClassifier Without
#Train the model using winetype RandomForestClassifier

clf = RandomForestClassifier(max_depth=Depth, random_state=0, n_estimators = 500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
RFC_Without_report = classification_report(y_test, y_pred)
print("Classification Report RFC_Without:\n", RFC_Without_report)
#%% Logistic regression Without
log_reg = LogisticRegression(random_state=0, max_iter=25000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
LR_Without_report = classification_report(y_test, y_pred)
print("Classification Report LR_Without:\n", LR_Without_report)
#%% KNN Without
knn = KNeighborsClassifier(n_neighbors=Neighbours)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
KNN_Without_report = classification_report(y_test, y_pred)
print("Classification Report KNN_Without:\n", KNN_Without_report)

#%% Decision Tree Without
decision_tree = DecisionTreeClassifier(max_depth=Depth, random_state=0)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
DT_Without_report = classification_report(y_test, y_pred)
print("Classification Report DT_Without:\n", DT_Without_report)
