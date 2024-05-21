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
X_qc = df.drop(['quality', 'type', 'y'], axis=1).values
X_qc[:,11] = [1 if x == 'red' else 0 for x in X_qc[:,11]]
X_red = X_qc[X_qc[:,11] == 1][:, :11]
X_white = X_qc[X_qc[:,11] == 0][:, :11]
y_red = y_quality[:1599:]
y_white = y_quality[1599:]

#%%  Train and Test
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_color, test_size=0.2, random_state=123)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_type, test_size=0.2, random_state=123)
X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X, y_quality, test_size=0.2, random_state=123)
X_train_qc, X_test_qc, y_train_qc, y_test_qc = train_test_split(X_qc, y_quality, test_size=0.2, random_state=123)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.2, random_state=123)
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.2, random_state=123)

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
        
        optimal[name] = {'model': None, 'params': {}}
        
        for key in parameters.keys():
            if name in key:
                print(key)
                cv = GridSearchCV(pipeline, param_grid={key : parameters[key]}, scoring = 'recall_macro')
                cv.fit(X_tr, y_tr)
                y_predict = cv.predict(X_te)

                print(cv.best_score_)
                print(cv.best_params_)
                
                test_score = cv.score(X_te, y_te)
                # print('{} test set Accuracy: {}'.format(name, test_score))
                print('{} test set Recall: {}'.format(name, test_score))
                
                optimal[name]['model'] = cv.best_estimator_
                optimal[name]['params'].update(cv.best_params_)
                
    return optimal

#%%
optimal_t = BestParams(X_train_t, X_test_t, y_train_t, y_test_t)
optimal_c = BestParams(X_train_c, X_test_c, y_train_c, y_test_c)
optimal_q = BestParams(X_train_q, X_test_q, y_train_q, y_test_q)
optimal_qc = BestParams(X_train_qc, X_test_qc, y_train_qc, y_test_qc)
optimal_red = BestParams(X_train_red, X_test_red, y_train_red, y_test_red)
optimal_white = BestParams(X_train_white, X_test_white, y_train_white, y_test_white)

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
            options[name]['model'] = pipeline
        else:
            pipeline.fit(X_tr, y_tr)
            pipeline.score(X_tr, y_tr)
            options[name]['y'] = pipeline.predict(X_te)
            options[name]['model'] = pipeline
            
        cv_result = cross_val_score(pipeline, X_tr, y_tr, cv=kf)
        results.append(cv_result)
        
    plt.boxplot(results, labels=models.keys())
    plt.xticks(rotation = 45)
    plt.title(title)
    plt.show()
        
    return results
        
#%%            
results_t = RunModels(X_train_t, X_test_t, y_train_t, y_test_t, optimal_t, 'Wine Type: great, good, poor')
results_c = RunModels(X_train_c, X_test_c, y_train_c, y_test_c, optimal_c, 'Wine Color')
results_q = RunModels(X_train_q, X_test_q, y_train_q, y_test_q, optimal_q, 'Wine Quality: great, bad')
results_qc = RunModels(X_train_qc, X_test_qc, y_train_qc, y_test_qc, optimal_qc, 'Wine Quality with color: great, bad')
results_rw = RunModels(X_train_red, X_test_white, y_train_red, y_test_white, optimal_red, 'Red Model: White Wine Quality')
results_wr = RunModels(X_train_white, X_test_red, y_train_white, y_test_red, optimal_white, 'White Model: Red Wine Quality')
results_red = RunModels(X_train_red, X_test_red, y_train_red, y_test_red, optimal_red, 'Red Wine Quality: great, bad')
results_white = RunModels(X_train_white, X_test_white, y_train_white, y_test_white, optimal_white, 'White Wine Quality: great, bad')

#%%
def CrosTable(y_labels, y_data):
    df_check = pd.DataFrame({'predicted': y_labels, 'original': y_data})
    ct = pd.crosstab(df_check['predicted'], df_check['original'])
    print(ct)
    return ct

#%%
def GetScores(y_te, y_pr):
    accuracy = accuracy_score(y_te, y_pr)
    precision = precision_score(y_te, y_pr, average = "macro")
    recall = recall_score(y_te, y_pr, average = "macro")
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
df_scores_qc = {}
df_scores_red = {}
df_scores_white = {}
df_scores_rw = {}
df_scores_wr = {}
index = ["Accuracy", "Precision", "Recall", "F1", "Jaccard", "Average"]

for name in models.keys():
    CrosTable(optimal_t[name]['y'], y_test_t)
    cm_t, df_scores_t[name] = GetScores(y_test_t, optimal_t[name]['y'])
    CrosTable(optimal_c[name]['y'], y_test_c)
    cm_c, df_scores_c[name] = GetScores(y_test_c, optimal_c[name]['y'])
    CrosTable(optimal_q[name]['y'], y_test_q)
    cm_q, df_scores_q[name] = GetScores(y_test_q, optimal_q[name]['y'])
    
    CrosTable(optimal_qc[name]['y'], y_test_qc)
    cm_qc, df_scores_qc[name] = GetScores(y_test_qc, optimal_qc[name]['y'])
    
    CrosTable(optimal_red[name]['y'], y_test_red)
    cm_red, df_scores_red[name] = GetScores(y_test_red, optimal_red[name]['y'])
    CrosTable(optimal_white[name]['y'], y_test_white)
    cm_white, df_scores_white[name] = GetScores(y_test_white, optimal_white[name]['y'])
    
    # Red Wine Model vs White Wine Data
    model_red = optimal_red[name]['model']
    y_pred_rw = model_red.predict(X_test_white)
    CrosTable(y_pred_rw, y_test_white)
    cm_rw, df_scores_rw[name] = GetScores(y_test_white, y_pred_rw)
    # White Wine Model vs Red Wine Data
    model_white = optimal_white[name]['model']
    y_pred_wr = model_white.predict(X_test_red)
    CrosTable(y_pred_wr, y_test_red)
    cm_wr, df_scores_wr[name] = GetScores(y_test_red, y_pred_wr)

df_scores_t = pd.DataFrame(df_scores_t, index=index)
print(df_scores_t)
df_scores_c = pd.DataFrame(df_scores_c, index=index)
print(df_scores_c)
df_scores_q = pd.DataFrame(df_scores_q, index=index)
print(df_scores_q)

df_scores_qc = pd.DataFrame(df_scores_qc, index=index)
print(df_scores_qc)
df_scores_red = pd.DataFrame(df_scores_red, index=index)
print(df_scores_red)
df_scores_white = pd.DataFrame(df_scores_white, index=index)
print(df_scores_white)

df_scores_rw = pd.DataFrame(df_scores_rw, index=index)
print(df_scores_rw)
df_scores_wr = pd.DataFrame(df_scores_wr, index=index)
print(df_scores_wr)

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

colors = ['#2B88BD' if value == 'great' else '#EB9D3F' if value == 'poor' else '#B5B1B1' for value in df_pca_t['RandomForest']]
plt.scatter(df_pca_t['pc_1'], df_pca_t['pc_2'], c=colors, alpha=0.6) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest for Type: great, good, poor')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Good', markerfacecolor='#B5B1B1', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#2B88BD', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Poor', markerfacecolor='#EB9D3F', markersize=8)
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
plt.title('SVC for Color')
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

colors = ['#2B88BD' if value == 1 else '#EB9D3F' for value in df_pca_q['RandomForest']]
plt.scatter(df_pca_q['pc_1'], df_pca_q['pc_2'], c=colors, alpha=0.6) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest for Quality: great, bad')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#2B88BD', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#EB9D3F', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Quality and Color
scaled_qc = scaler.fit_transform(X_test_qc)
pc_qc = pca2.fit_transform(scaled_qc)

colors = ['#2B88BD' if value == 1 else '#EB9D3F' for value in optimal_qc['RandomForest']['y']]
plt.scatter(pc_qc[:, 0], pc_qc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest Quality with color data: great, bad')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#2B88BD', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#EB9D3F', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Quality Red Wine
scaled_red = scaler.fit_transform(X_test_red)
pc_red = pca2.fit_transform(scaled_red)

colors = ['#722f37' if value == 1 else '#B5B1B1' for value in optimal_red['RandomForest']['y']]
plt.scatter(pc_red[:, 0], pc_red[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest Red Wine Quality')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#722f37', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#B5B1B1', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Quality White Wine
scaled_white = scaler.fit_transform(X_test_white)
pc_white = pca2.fit_transform(scaled_white)

colors = ['#f9e8c0' if value == 1 else '#B5B1B1' for value in optimal_white['RandomForest']['y']]
plt.scatter(pc_white[:, 0], pc_white[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('RandomForest White Wine Quality')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#f9e8c0', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#B5B1B1', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Quality Red Model vs White Wine
model_red = optimal_red['NET']['model']
y_pred_rw = model_white.predict(X_test_white)

colors = ['#f9e8c0' if value == 1 else '#B5B1B1' for value in y_pred_rw]
plt.scatter(pc_white[:, 0], pc_white[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('NET Red Model vs White Wine Data')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#f9e8c0', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#B5B1B1', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% PCA Quality White Model vs Red Wine
model_white = optimal_white['SVC']['model']
y_pred_wr = model_white.predict(X_test_red)

colors = ['#722f37' if value == 1 else '#B5B1B1' for value in y_pred_wr]
plt.scatter(pc_red[:, 0], pc_red[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('SVC White Model vs Red Wine Data')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#722f37', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#B5B1B1', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

#%% Whole Data
scaled = scaler.fit_transform(X)
pc = pca2.fit_transform(scaled)

# TYPE
colors = ['#2B88BD' if value == 'great' else '#EB9D3F' if value == 'poor' else '#B5B1B1' for value in y_type]
plt.scatter(pc[:, 0], pc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Type: great, good, poor')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Good', markerfacecolor='#B5B1B1', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#2B88BD', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Poor', markerfacecolor='#EB9D3F', markersize=8)
]
plt.legend(handles=legend_elements)
plt.show()

# COLOR
colors = ['#722f37' if value == 'red' else '#f9e8c0' for value in y_color]
plt.scatter(pc[:, 0], pc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Color: red, white')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Red', markerfacecolor='#722f37', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='White', markerfacecolor='#f9e8c0', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()

# QUALITY
colors = ['#2B88BD' if value == 1 else '#EB9D3F' for value in y_quality]
plt.scatter(pc[:, 0], pc[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Quality: great, bad')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Great', markerfacecolor='#2B88BD', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='#EB9D3F', markersize=8),
]
plt.legend(handles=legend_elements)
plt.show()


#%%
