# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:21:53 2024

@author: BeButton
"""


#%%  Packages

import random
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, jaccard_score

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.decomposition import PCA

#%%   Data
random.seed(123)
red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv', sep=';')

#%%
print(red.columns)
print(white.columns)

#%%
print(red.groupby('quality').sum())
print(white.groupby('quality').sum())

#%%
red['color'] = 'red'
white['color'] = 'white'

#%%
df = pd.concat([red, white], axis=0)

#%%   Labels
df['type'] = ['great' if x > 6 else 'poor' if x < 5 else 'good' for x in df.quality]
df['y'] = [1 if x > 6 else 0 for x in df.quality]

#%%
X = df.drop(['quality', 'color', 'type', 'y'], axis=1).values
y_color = df['color'].values
y_type = df['type'].values
y_quality = df['y'].values

#%%  Train and Test
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_color, test_size=0.2, random_state=123)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_type, test_size=0.2, random_state=123)
X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X, y_quality, test_size=0.2, random_state=123)

#%%
models = {'KNN' : KNeighborsClassifier(),
          'DecisionTree' : DecisionTreeClassifier(random_state=24),
          'NaiveBayes' : GaussianNB(),
          'NET' : MLPClassifier(random_state=24),
          'SVC' : SVC(random_state=24),
          'LogReg' : LogisticRegression(random_state=24), 
          'Boost' : AdaBoostClassifier(random_state=24),
          'RandomForest' : RandomForestClassifier(random_state=24)}

parameters = {'KNN__n_neighbors': np.arange(1, 50),
              'DecisionTree__criterion': ['gini', 'entropy', 'log_loss'],
              'NET__hidden_layer_sizes': np.arange(50, 201, 50),
              'SVC__kernel' :  ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              'SVC__C' :  np.arange(1, 10),
              'LogReg__C' :  np.arange(1, 10),
              'RandomForest__criterion': ['gini', 'entropy', 'log_loss']}

#%%
def BestParams(X_tr, X_te, y_tr, y_te):
    optimal = {}
    for name, model in models.items():
        print(name)
        steps = [('scaler', StandardScaler()), (name, model)]
        pipeline = Pipeline(steps)
        
        optimal[name] = {'model': model, 'params': {}}
        
        for key in parameters.keys():
            if name in key:
                print(key)
                cv = GridSearchCV(pipeline, param_grid={key : parameters[key]})
                cv.fit(X_tr, y_tr)
                y_predict = cv.predict(X_te)

                print(cv.best_score_)
                print(cv.best_params_)
                
                test_score = cv.score(X_te, y_te)
                print('{} test set Accuracy: {}'.format(name, test_score))
                
                optimal[name]['params'].update(cv.best_params_)
                
    return optimal

#%%
optimal_t = BestParams(X_train_t, X_test_t, y_train_t, y_test_t)
optimal_c = BestParams(X_train_c, X_test_c, y_train_c, y_test_c)
optimal_q = BestParams(X_train_q, X_test_q, y_train_q, y_test_q)

#%%    
def RunModels(X_tr, X_te, y_tr, y_te, options, title):
    results = []     
           
    for name, model in models.items():
        print(name)
        steps = [('scaler', StandardScaler()), (name, model)]
        kf = KFold(n_splits=4, random_state=24, shuffle=True)
        pipeline = Pipeline(steps)
        # print(model.get_params().keys())
        
        if name in options.keys():
            param = options[name]['params']
            pipeline.set_params(**param)
            pipeline.fit(X_tr, y_tr)
            pipeline.score(X_tr, y_tr)
            options[name]['y'] = pipeline.predict(X_te)
        else:
            pipeline.fit(X_tr, y_tr)
            pipeline.score(X_tr, y_tr)
            options[name]['y'] = pipeline.predict(X_te)
            
        cv_result = cross_val_score(pipeline, X_tr, y_tr, cv=kf)
        results.append(cv_result)
        
    plt.boxplot(results, labels=models.keys())
    plt.xticks(rotation = 45)
    plt.title(title)
    plt.show()
        
    return results
        
#%%            
results_t = RunModels(X_train_t, X_test_t, y_train_t, y_test_t, optimal_t, 'Wine Type')
results_c = RunModels(X_train_c, X_test_c, y_train_c, y_test_c, optimal_c, 'Wine Color')
results_q = RunModels(X_train_q, X_test_q, y_train_q, y_test_q, optimal_q, 'Wine Quality')

#%%
def CrosTable(y_labels, y_data):
    df_check = pd.DataFrame({'predicted': y_labels, 'original': y_data})
    ct = pd.crosstab(df_check['predicted'], df_check['original'])
    print(ct)
    return ct

#%%
def GetScores(y_te, y_pr):
    precision = precision_score(y_te, y_pr, average = "macro")
    recall = recall_score(y_te, y_pr, average = "macro")
    accuracy = accuracy_score(y_te, y_pr)
    f1 = f1_score(y_te, y_pr, average = "macro")
    jaccard = jaccard_score(y_te, y_pr, average = "macro")
    
    cm = confusion_matrix(y_te, y_pr)
    cm_total = {'TN': cm[0, 0], 'FP': cm[0, 1],
            'FN': cm[1, 0], 'TP': cm[1, 1]}
    
    scores = [accuracy, precision, recall, f1, jaccard]
    scores.append(np.mean(scores))
    
    return cm_total, scores

#%%
df_scores_t = {}
df_scores_c = {}
df_scores_q = {}
index = ["Accuracy", "Precision", "Recall", "F1", "Jaccard", "Average"]

for name in models.keys():
    CrosTable(optimal_t[name]['y'], y_test_t)
    cm_t, df_scores_t[name] = GetScores(y_test_t, optimal_t[name]['y'])
    CrosTable(optimal_c[name]['y'], y_test_c)
    cm_c, df_scores_c[name] = GetScores(y_test_c, optimal_c[name]['y'])
    CrosTable(optimal_q[name]['y'], y_test_q)
    cm_q, df_scores_q[name] = GetScores(y_test_q, optimal_q[name]['y'])

df_scores_t = pd.DataFrame(df_scores_t, index=index)
print(df_scores_t)
df_scores_c = pd.DataFrame(df_scores_c, index=index)
print(df_scores_c)
df_scores_q = pd.DataFrame(df_scores_q, index=index)
print(df_scores_q)

#%% PCA
scaler = StandardScaler()
scaled_df = scaler.fit_transform(X)
pca = PCA(n_components=6)
pca.fit_transform(scaled_df)

print(pca.components_)
print(sum(pca.explained_variance_ratio_))

nums = np.arange(12)
 
var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(scaled_df)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

#%% 
pca2 = PCA(n_components=2)

#%% PCA Type
scaled_t = scaler.fit_transform(X_test_t)
pc_t = pca2.fit_transform(scaled_t)

df_pca_t = pd.DataFrame(data = pc_t, columns = ["pc_1", "pc_2"])
df_pca_t['RandomForest'] = optimal_t['RandomForest']['y']

colors = ['blue' if value == 'great' else 'red' if value == 'poor' else 'yellow' for value in df_pca_t['RandomForest']]
plt.scatter(df_pca_t['pc_1'], df_pca_t['pc_2'], c=colors, alpha=0.6) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest for Type Classification')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Good', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Poor', markerfacecolor='red', markersize=8)
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Color
scaled_c = scaler.fit_transform(X_test_c)
pc_c = pca2.fit_transform(scaled_c)

df_pca_c = pd.DataFrame(data = pc_c, columns = ["pc_1", "pc_2"])
df_pca_c['SVC'] = optimal_c['SVC']['y']

colors = ['#722f37' if value == 'red' else '#f9e8c0' for value in df_pca_c['SVC']]
plt.scatter(df_pca_c['pc_1'], df_pca_c['pc_2'], c=colors, alpha=0.6) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('SVC for Color Classification')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Red', markerfacecolor='#722f37', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='White', markerfacecolor='#f9e8c0', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Quality
scaled_q = scaler.fit_transform(X_test_q)
pc_q = pca2.fit_transform(scaled_q)

df_pca_q = pd.DataFrame(data = pc_q, columns = ["pc_1", "pc_2"])
df_pca_q['RandomForest'] = optimal_q['RandomForest']['y']

colors = ['green' if value == 1 else 'red' for value in df_pca_q['RandomForest']]
plt.scatter(df_pca_q['pc_1'], df_pca_q['pc_2'], c=colors, alpha=0.6) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest for Quality Classification')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='green', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='red', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% Whole Data
scaled = scaler.fit_transform(X)
pc = pca2.fit_transform(scaled)

# TYPE
colors = ['blue' if value == 'great' else 'red' if value == 'poor' else 'yellow' for value in y_type]
plt.scatter(pc[:, 0], pc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Type')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Good', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Poor', markerfacecolor='red', markersize=8)
]
plt.legend(handles=legend_elements)
plt.show()

# COLOR
colors = ['#722f37' if value == 'red' else '#f9e8c0' for value in y_color]
plt.scatter(pc[:, 0], pc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Color')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Red', markerfacecolor='#722f37', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='White', markerfacecolor='#f9e8c0', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

# QUALITY
colors = ['green' if value == 1 else 'red' for value in y_quality]
plt.scatter(pc[:, 0], pc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Quality')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='green', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='red', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()












