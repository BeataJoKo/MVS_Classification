# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:49:27 2024

@author: Andreas Jørgensen
"""

#%% Step 1, Importing Packages for Data mining and Machine Learning project on Clasification of Wine Data sets. 
import os 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as st
import seaborn as sns  
import numpy as np
import pandas as pd 
import sklearn as skl 

#%% Step 2, Loading the Wine Data Sets for both Red & White as csv (comma seperated values) files 

# Red wine Data set 
red_wine_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Data Mining & ML\winequality-red.csv")
red_wine_data['type'] = '1' 
# 1 is for red 
# White wine Data set 
white_wine_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Data Mining & ML\winequality-white.csv", delimiter = ';', quotechar = '"') 
white_wine_data['type'] = '2'
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

#%% Normalizing the Data and Removing outliers 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
normal_merged_wine_df = pd.DataFrame(scaler.fit_transform(merged_wine_df), columns = merged_wine_df.columns) 

#%% Outlier removal
def remove_outliers(df, z_threshold=3):
    from scipy.stats import zscore
    
    # Calculate the Z-scores
    z_scores = zscore(normal_merged_wine_df)
    
    # Create a boolean mask for rows with all Z-scores less than the threshold
    mask = (abs(z_scores) < z_threshold).all(axis=1)
    
    # Apply the mask to filter out outliers
    normal_merged_wine_df_no_outliers = normal_merged_wine_df[mask]
    
    return normal_merged_wine_df_no_outliers

# Apply the function to remove outliers from the normalized data
cleaned_wine_df = remove_outliers(normal_merged_wine_df)

#%% Normality check 
norm_check = cleaned_wine_df.columns.tolist()
plt.figure(figsize=(15,11))
for i in range(0, len(norm_check)):
  plt.subplot(3, 4, i+1)
  sns.boxplot(x=cleaned_wine_df[norm_check[i]], color='green', orient='h')
plt.suptitle('Boxplot Distribution of All Collumns')



#%% Label construction

# adding the "Ratings" varialbe to the merged data sets atributes
#cleaned_wine_df['rating'] = ['great' if x > 7 else 'poor' if x < 5 else 'good' for x in cleaned_wine_df.quality] 
# adding the binary clasifier column "y" as the outputs "Class label"
#cleaned_wine_df['y-Class label'] = [1 if x > 6 else 0 for x in cleaned_wine_df.quality] 
cleaned_wine_df= cleaned_wine_df.copy()
cleaned_wine_df['rating'] = cleaned_wine_df['rating'].apply(
    lambda x: 'great' if x > 7 else 'poor' if x < 5 else 'good')


print(" The Merged Data sets for both Red & White wine")
print(cleaned_wine_df) 
Training_set_df = cleaned_wine_df
#%% Boxplot distribution for the merged data
features = Training_set_df.columns.tolist()
plt.figure(figsize=(15,11))
for i in range(0, len(features)):
  plt.subplot(3, 4, i+1)
  sns.boxplot(x=Training_set_df[features[i]], color='green', orient='h')
plt.suptitle('Boxplot Distribution of All Collumns')

#%% Histogram plot distribution
no_target = ['fixed acidity',
  'volatile acidity',
  'citric acid',
  'residual sugar',
  'chlorides',
  'free sulfur dioxide',
  'total sulfur dioxide',
  'density',
  'pH',
  'sulphates',
  'alcohol']
plt.figure(figsize=(15,15))
for i in range(0, len(no_target)):
  plt.subplot(3, 4, i+1)
  ax = sns.histplot(x= Training_set_df[no_target[i]], color='green')
  
#D'Agostino-Pearson Normality Test
print("Normality test for the Features")
for i in range(len(no_target)):
  norm = st.normaltest(np.array(Training_set_df[no_target[i]]))
  if norm.pvalue > 0.05:
    print('p-Value of', no_target[i], 'is :', norm.pvalue, 'can be categorize as Normal')
  else:
    print('p-Value of', no_target[i], 'is :', norm.pvalue, 'can be categorize as Not Normal')
print("The p-values for all the features are not normally distributed, therefor they need to be stanardized and outliers should be removed")    

# Traing split of the Merged_wine_df (80%) 

# Test split of the Merged_wine_df (20%)
#%% Distribution for ratings "Histogram" 
ax = sns.histplot(x=Training_set_df['rating'], color='blue')
labels = [str(v) if v else '' for v in ax.containers[0].datavalues]
ax.bar_label(ax.containers[0], labels=labels)
plt.xlabel('Quality Rating')
plt.title('Quality Ratings Histogram Distribution')
plt.show() 
#%% Histogram for ratings vs quality 
ax = sns.barplot(x=Training_set_df['rating'], y=Training_set_df['quality'], estimator='mean', color='red', errorbar=None)
ax.bar_label(ax.containers[0], fmt='%.2f')
plt.title('Average rating vs quality')
plt.show()

#%% Multivar on Train Data  
plt.figure(figsize=(10,10))
ax = sns.heatmap(data=Training_set_df.corr(method='spearman'), annot= True, cmap='Blues', fmt='.2f')
ax.add_patch(patches.Rectangle((0,11),13,1,edgecolor='red',lw=3)) 
print("The Data does not seem to indicate any significant relationship between quality and characteristics") 

#%% Seting up the Training & Test Sets for Classification from the Merged Wine Data Frames 
# Aim is to Randomize the observations of the Merged Red & White wine data 
Training_set_df = merged_wine_df.sample(frac = 1) 
print("The Ramdomized Set of wine Data for Training is as follows:")
print(Training_set_df)
print("The Descriptive Statistics for the Training set are as follows:")
print(Training_set_df.describe())