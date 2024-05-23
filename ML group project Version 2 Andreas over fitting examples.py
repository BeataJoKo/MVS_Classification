# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:49:27 2024

@author: Andreas Jørgensen
"""

#%% Step 1, Importing Packages for Data mining and Machine Learning project on Clasification of Wine Data sets. 
import os 
from scipy import stats
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as st
import seaborn as sns  
import numpy as np
import pandas as pd 
import sklearn as skl 
# import scikit_posthocs as sp
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, GridSearchCV, RandomizedSearchCV
from imblearn import under_sampling, over_sampling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import shap
import pickle
#%% Step 2, Loading the Wine Data Sets for both Red & White as csv (comma seperated values) files 

# Red wine Data set 
red_wine_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Data Mining & ML\winequality-red.csv")
red_wine_data['type'] = '0' 
# 1 is for red 
# White wine Data set 
white_wine_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Data Mining & ML\winequality-white.csv", delimiter = ';', quotechar = '"') 
white_wine_data['type'] = '1'
# 2 is for white

#print(red_wine_data) 

#print(white_wine_data) 

#%% Step 3, Data Exploration for Red wine data set
red_df = pd.DataFrame(red_wine_data) 
print("First few rows for the red wine Dataframe:")
print(red_df.head()) 
print("The data type for the Red wine Data set are as follows:")
print(red_df.dtypes)
print("Analysis of is missing values:")
print(red_df.isnull().sum())
print("The summary Statistics for the Red wine data set is as follows:")
print(red_df.describe())
# Box plot for red wine dataset
red_df.hist(figsize=(12, 10))
plt.suptitle("Histograms for Red Wine Dataset")
plt.show()
#%% Step 4, Data Exploration for White wine data set
white_df = pd.DataFrame(white_wine_data)
print("First few rows for the White wine Dataframe:")
print(white_df.head()) 
print("The data type for the White wine Data set are as follows:")
print(white_df.dtypes)
print("Analysis of is missing values:")
print(white_df.isnull().sum())
print("The summary Statistics for the White wine data set is as follows:")
print(white_df.describe())

# Histogram for white wine dataset
white_df.hist(figsize=(12, 10))
plt.suptitle("Histograms for White Wine Dataset")
plt.show()

#%% Step 5, Merging the red & white wine Data sets together
data_frames = [red_df, white_df] 
merged_wine_df = pd.concat(data_frames) 

#%% Checking for any duplicate values  
print("Checking for any duplicate Values")
print("Are there any duplicate values in the data set :", merged_wine_df.duplicated().any())
print("How many duplicate values are there :", merged_wine_df.duplicated().sum())
duplicated_rows = merged_wine_df[merged_wine_df.duplicated(keep = False) == True]
print(duplicated_rows)

#%% Removing the duplicate values in the merged data 
merged_wine_df = merged_wine_df.drop_duplicates()
print("Are there remaining duplicate values: ", merged_wine_df.duplicated().any())
print("How many :", merged_wine_df.duplicated().sum())



#%% Label construction

# adding the "Ratings" varialbe to the merged data sets atributes
#cleaned_wine_df['rating'] = ['great' if x > 7 else 'poor' if x < 5 else 'good' for x in cleaned_wine_df.quality] 
# adding the binary clasifier column "y" as the outputs "Class label"
#cleaned_wine_df['y-Class label'] = [1 if x > 6 else 0 for x in cleaned_wine_df.quality] 
merged_wine_df['type'] = ['great' if x > 6 else 'poor' if x < 5 else 'good' for x in merged_wine_df.quality]
merged_wine_df['y'] = [1 if x > 6 else 0 for x in merged_wine_df.quality]

print(" The Merged Data sets for both Red & White wine")
print(merged_wine_df) 
Training_set_df = merged_wine_df

#%% Distribution for ratings "Histogram" 
ax = sns.histplot(x=Training_set_df['type'], color='blue')
labels = [str(v) if v else '' for v in ax.containers[0].datavalues]
ax.bar_label(ax.containers[0], labels=labels)
plt.xlabel('Quality Rating')
plt.title('Quality Ratings Histogram Distribution')
plt.show() 
#%% Histogram for ratings vs quality 
ax = sns.barplot(x=Training_set_df['type'], y=Training_set_df['quality'], estimator='mean', color='red', errorbar=None)
ax.bar_label(ax.containers[0], fmt='%.2f')
plt.title('Average rating vs quality')
plt.show()

#%% Seting up the Training & Test Sets for Classification from the Merged Wine Data Frames 
# Aim is to Randomize the observations of the Merged Red & White wine data 
Training_set_df = merged_wine_df.sample(frac = 1) 
print("The Ramdomized Set of wine Data for Training is as follows:")
print(Training_set_df)
print("The Descriptive Statistics for the Training set are as follows:")
print(Training_set_df.describe()) 

#%% Spliting the Data for Training and Testing 
inputs = Training_set_df.drop('type', axis = 1)
target = Training_set_df['quality']
X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size = 0.2, random_state=17)
# Chainging the qualitiy into a binary clisificatiob condition
Y_train= Y_train.apply(lambda x: 0 if x <= 6 else 1)
Y_test = Y_test.apply(lambda x: 0 if x <= 6 else 1)
Y_train.value_counts()

labels = ['Bad','Good']
plt.pie(x = Y_train.value_counts(), labels = labels, colors = ['red','green'], autopct = '%.2f')
plt.title('Proportions of Quality')
plt.show()

#%% Modeling Perfomance indicators ie. Precission, Recall, Accuracy, ROC/AUC, Cross Validation
# create function to show the metrics results for each model
def eval_metrics(y_train, y_test, y_pred_train, y_pred_test):
  #store the metrics
  precision_train = precision_score(Y_train, Y_pred_train)
  precision_test = precision_score(Y_test, Y_pred_test)
  recall_train = recall_score(Y_train, Y_pred_train)
  recall_test = recall_score(Y_test, Y_pred_test)
  accuracy_train = accuracy_score(Y_train, Y_pred_train)
  accuracy_test = accuracy_score(Y_test, Y_pred_test)
  roc_train = roc_auc_score(Y_train, Y_pred_train)
  roc_test = roc_auc_score(Y_test, Y_pred_test) 
  
  #show the score
  print('precision train score:', precision_train)
  print('precision test score:', precision_test)
  print('recall train score:', recall_train)
  print('recall test score:', recall_test)
  print('accuracy train score:', accuracy_train)
  print('accuracy test score:', accuracy_test)
  print('roc auc train score:', roc_train)
  print('roc auc test score:', roc_test)
  #show the confusion matrix
  cm_test = confusion_matrix(Y_test, Y_pred_test)
  cm_display = ConfusionMatrixDisplay(cm_test, display_labels=['Bad (0)','Good (1)'])
  cm_display.plot(cmap='Blues',colorbar=False)
  plt.grid(None)
  plt.show()

  #store into list
  result_train = [precision_train, recall_train, accuracy_train, roc_train]
  result_test = [precision_test, recall_test, accuracy_test, roc_test]
  return result_train, result_test
  
#%% Create function contains cross validation for every model
metrics_list = ['recall', 'precision', 'accuracy', 'roc_auc']
def cv_metrics(model):
  cv = cross_validate(model, X_train, Y_train, scoring=metrics_list, cv=5, return_train_score=True)
  print('Standard deviation of precision train :', cv['train_precision'].std())
  print('Standard deviation of precision test :', cv['test_precision'].std())
  print('Standard deviation of recall train :', cv['train_recall'].std())
  print('Standard deviation of recall test :', cv['test_recall'].std())
  print('Standard deviation of accuracy train :', cv['train_accuracy'].std())
  print('Standard deviation of accuracy test :', cv['test_accuracy'].std())
  print('Standard deviation of roc_auc train :', cv['train_roc_auc'].std())
  print('Standard deviation of roc_auc test :', cv['test_roc_auc'].std())
  #Store into list
  cv_train = [cv['train_precision'].std(), cv['train_recall'].std(), cv['train_accuracy'].std(), cv['train_roc_auc'].std()]
  cv_test = [cv['test_precision'].std(), cv['test_recall'].std(), cv['test_accuracy'].std(), cv['test_roc_auc'].std()]
  return cv_train, cv_test

#%% Logistic Regression 
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels on Y_train and Y_test
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

scaler = StandardScaler()
reg = LogisticRegression(random_state=17)
pipe_reg = Pipeline([('transform', scaler), ('model', reg)])
pipe_reg.fit(X_train, Y_train_encoded)
Y_pred_train = pipe_reg.predict_proba(X_train)[:, 1]
Y_pred_test = pipe_reg.predict_proba(X_test)[:, 1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test] 
result_test_reg = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)

cv_train_reg, cv_test_reg = cv_metrics(pipe_reg)

#%% Support Vector Machine 

svm = SVC(probability=True, random_state=17)
pipe_svm = Pipeline([('transform' , scaler), ('model', svm)])
pipe_svm.fit(X_train, Y_train)
Y_pred_train = pipe_svm.predict_proba(X_train)[:,1]
Y_pred_test = pipe_svm.predict_proba(X_test)[:,1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_svm, result_test_svm = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)

cv_train_svm, cv_test_svm = cv_metrics(pipe_svm) 

#%% Decision Tree 

dtc = DecisionTreeClassifier(random_state=17)
pipe_dtc = Pipeline([('transform' , scaler), ('model', dtc)])
pipe_dtc.fit(X_train, Y_train)
Y_pred_train = pipe_dtc.predict_proba(X_train)[:,1]
Y_pred_test = pipe_dtc.predict_proba(X_test)[:,1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_dtc, result_test_dtc = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test) 

cv_train_dtc, cv_test_dtc = cv_metrics(pipe_dtc) 

#%% Extreme Gradient Boost (XGBoost) 

xgb = XGBClassifier(random_state=17)
pipe_xgb = Pipeline([('transform' , scaler), ('model', dtc)])
pipe_xgb.fit(X_train, Y_train)
Y_pred_train = pipe_xgb.predict_proba(X_train)[:,1]
Y_pred_test = pipe_xgb.predict_proba(X_test)[:,1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_xgb, result_test_xgb = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test) 

cv_train_xgb, cv_test_xgb = cv_metrics(pipe_xgb) 

#%% Random Forest Classifier 

rfc = RandomForestClassifier(random_state=17)
pipe_rfc = Pipeline([('transform' , scaler), ('model', dtc)])
pipe_rfc.fit(X_train, Y_train)
Y_pred_train = pipe_rfc.predict_proba(X_train)[:,1]
Y_pred_test = pipe_rfc.predict_proba(X_test)[:,1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_rfc, result_test_rfc = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test) 

cv_train_rfc, cv_test_rfc = cv_metrics(rfc)

#%% fixed portion of code. 

# Define the evaluation metrics function
def eval_metrics(y_train, y_test, y_pred_train, y_pred_test):
    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)
    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    roc_train = roc_auc_score(y_train, y_pred_train)
    roc_test = roc_auc_score(y_test, y_pred_test)

    print('Precision train score:', precision_train)
    print('Precision test score:', precision_test)
    print('Recall train score:', recall_train)
    print('Recall test score:', recall_test)
    print('Accuracy train score:', accuracy_train)
    print('Accuracy test score:', accuracy_test)
    print('ROC AUC train score:', roc_train)
    print('ROC AUC test score:', roc_test)

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_display = ConfusionMatrixDisplay(cm_test, display_labels=['Bad (0)', 'Good (1)'])
    cm_display.plot(cmap='Blues', colorbar=False)
    plt.grid(None)
    plt.show()

    result_train = [precision_train, recall_train, accuracy_train, roc_train]
    result_test = [precision_test, recall_test, accuracy_test, roc_test]
    return result_train, result_test

# Define cross-validation metrics function
def cv_metrics(model):
    cv = cross_validate(model, X_train, Y_train_encoded, scoring=metrics_list, cv=5, return_train_score=True)
    print('Standard deviation of precision train:', cv['train_precision'].std())
    print('Standard deviation of precision test:', cv['test_precision'].std())
    print('Standard deviation of recall train:', cv['train_recall'].std())
    print('Standard deviation of recall test:', cv['test_recall'].std())
    print('Standard deviation of accuracy train:', cv['train_accuracy'].std())
    print('Standard deviation of accuracy test:', cv['test_accuracy'].std())
    print('Standard deviation of roc_auc train:', cv['train_roc_auc'].std())
    print('Standard deviation of roc_auc test:', cv['test_roc_auc'].std())

    cv_train = [cv['train_precision'].std(), cv['train_recall'].std(), cv['train_accuracy'].std(), cv['train_roc_auc'].std()]
    cv_test = [cv['test_precision'].std(), cv['test_recall'].std(), cv['test_accuracy'].std(), cv['test_roc_auc'].std()]
    return cv_train, cv_test

# Fit Logistic Regression model
pipe_reg.fit(X_train, Y_train_encoded)
Y_pred_train = pipe_reg.predict_proba(X_train)[:, 1]
Y_pred_test = pipe_reg.predict_proba(X_test)[:, 1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_test_reg = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)
cv_train_reg, cv_test_reg = cv_metrics(pipe_reg)

# Fit Support Vector Machine model
pipe_svm.fit(X_train, Y_train_encoded)
Y_pred_train = pipe_svm.predict_proba(X_train)[:, 1]
Y_pred_test = pipe_svm.predict_proba(X_test)[:, 1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_svm, result_test_svm = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)
cv_train_svm, cv_test_svm = cv_metrics(pipe_svm)

# Fit Decision Tree model
pipe_dtc.fit(X_train, Y_train_encoded)
Y_pred_train = pipe_dtc.predict_proba(X_train)[:, 1]
Y_pred_test = pipe_dtc.predict_proba(X_test)[:, 1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_dtc, result_test_dtc = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)
cv_train_dtc, cv_test_dtc = cv_metrics(pipe_dtc)

# Fit Extreme Gradient Boosting (XGBoost) model
pipe_xgb = Pipeline([('transform', scaler), ('model', xgb)])
pipe_xgb.fit(X_train, Y_train_encoded)
Y_pred_train = pipe_xgb.predict_proba(X_train)[:, 1]
Y_pred_test = pipe_xgb.predict_proba(X_test)[:, 1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_xgb, result_test_xgb = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)
cv_train_xgb, cv_test_xgb = cv_metrics(pipe_xgb)

# Fit Random Forest Classifier model
pipe_rfc = Pipeline([('transform', scaler), ('model', rfc)])
pipe_rfc.fit(X_train, Y_train_encoded)
Y_pred_train = pipe_rfc.predict_proba(X_train)[:, 1]
Y_pred_test = pipe_rfc.predict_proba(X_test)[:, 1]
Y_pred_train = [1 if i >= 0.75 else 0 for i in Y_pred_train]
Y_pred_test = [1 if i >= 0.75 else 0 for i in Y_pred_test]
result_train_rfc, result_test_rfc = eval_metrics(Y_train_encoded, Y_test_encoded, Y_pred_train, Y_pred_test)
cv_train_rfc, cv_test_rfc = cv_metrics(pipe_rfc)

# Feature Importance using SHAP
explainers = shap.explainers.Permutation(pipe_svm.predict, X_test, feature_names=X_train.columns.tolist())
shap_values = explainers(X_test)
shap.plots.beeswarm(shap_values)
