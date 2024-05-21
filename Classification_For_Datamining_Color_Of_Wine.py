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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%
df_white = pd.read_csv("winequality-white.csv", sep= ";")
df_white["Type"] = "White"
df_red= pd.read_csv("winequality-red.csv")
df_red["Type"] = "Red"

df = pd.concat([df_red,df_white])
describtion_red = df_red.describe()
describtion_white = df_white.describe()
#%%

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



df = df.drop(columns=["quality"]) 
# Features and target variable
X = df.drop(columns=["Type"])  # All columns except 'Type'
y = df["Type"]  # Target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%
#Supported vector machine classifier
svm_c = make_pipeline(StandardScaler(), LinearSVC())

svm_c.fit(X_train, y_train)
# Predict on the test set
y_pred = svm_c.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}" + " for SVM")
#Print classification report
print(classification_report(y_test, y_pred))
y_pred = svm_c.predict(X)
df["predicted"] = y_pred
#%%
#KNN classifier

knn_c = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_c.fit(X_train,y_train)
y_pred = knn_c.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}" + " for KNN")
#Print classification report
print(classification_report(y_test, y_pred))
#%%
#PCA


scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:,0:11]), index=df.iloc[:,0:11].index, columns=df.iloc[:,0:11].columns)



plt.figure(figsize=(20, 10))
sns.heatmap(df_scaled.corr(), annot=True, cmap='viridis')



pca = PCA(n_components=8)
pca.fit(df_scaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_[:2].sum())


# Visual for each component’s explained variance
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(pca.explained_variance_ratio_,"bo--",linewidth=2)
ax.set_xlabel("Principal Components", fontsize = 12)
ax.set_ylabel("Explained Variance", fontsize = 12)
ax.set_title("Explained Variance Ratio", fontsize = 16)
n = len(pca.explained_variance_ratio_)
plt.xticks(np.arange(n), np.arange(1, n+1));


# We see, that over 50% of the variance can be explained by just using two components

# Making a dataframe with principal components
pca2 = PCA(n_components=2)
pca2.fit(df_scaled)
principalComponents = pca2.fit_transform(df_scaled)

df_pca = pd.DataFrame(data = principalComponents, columns = ["principal component 1", "principal component 2"])
df = df.reset_index(drop=True)

df_pca.reset_index(inplace=True, drop=True)

df_pca = pd.concat(([df_pca, df["Type"]]), axis=1)
df_pca

colors = ['red' if value == "Red" else 'Yellow' for value in df_pca['Type']]
plt.scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1],c=colors)


red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Red Wine', markersize=10, linestyle='')
yellow_patch = plt.Line2D([0], [0], marker='o', color='yellow', label='White Wine', markersize=10, linestyle='')
plt.legend(handles=[red_patch, yellow_patch])

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title(' PC1 and PC2 as x and y axis')

# Show the plot
plt.show()
