＃清理資料－處理missing data的專案

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
insurance = pd.read_csv("data/lec03-insurance.csv") 

def split_train_test(data, test_ratio):
    np.random.seed(42)  
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(insurance, 0.2)
# print('\ntrain_set.info')
# train_set.info()
# print('\ntest_set.info')
# test_set.info()

## 看資料內容型態及概述
insurance.head()
insurance.describe()

## 4.1 Discover (發現) and Visualize (視覺化) the Data to Gain Insights (洞察)
# print(train_set.head(2))
insurance = train_set.copy()

## 4.1.1 Single variable (單變量)
insurance.charges.hist(bins = 3, figsize = (20,15))
plt.savefig('data//charges')
# plt.show()
# print(insurance.charges.min(), insurance.charges.max())
# print(insurance.charges.describe())

### 4.1.2 Looking for Correlations (尋找相關性)
corr_matrix = insurance.corr()
# print(corr_matrix) 
# print(corr_matrix["charges"].sort_values()) 
# print(corr_matrix["charges"].sort_values(ascending=False)) 

## 4.4 Prepare the Data for Machine Learning Algorithms (機器學習演算法)
## remove "charges" column
insurance = train_set.drop("charges", axis=1)
insurance_labels = train_set["charges"].copy()
# insurance.info()
# print(insurance.head(2))
# print(insurance_labels.head(2))
# print(insurance_labels.describe())

## 4.2.1 Dealing with missing Data
## (1) Delete
insurance5 = pd.read_excel('data/lec03-insurance-5.xlsx') 
# print(insurance5.head())
insurance5.dropna()
# insurance5.dropna().info()
insurance5.dropna(how = 'all')
# insurance5.dropna(how = 'all').info()

## (2) Replace with summary
insurance5.bmi 
# print(insurance5.bmi.mean())
# print(insurance5.sex.mode())
# print(insurance5.sex.mode()[0])
insurance5.head()
insurance5.fillna(insurance5.mode().iloc[0])
insurance5.head()
insurance5.fillna(insurance5.median()).fillna(insurance5.mode().iloc[0])

## (3) Random replace
insurance.bmi.describe()
insurance.bmi.min(), insurance.bmi.max()
insurance.bmi.min().round(), insurance.bmi.max().round()
random.randrange(insurance.bmi.min().round(), insurance.bmi.max().round())
insurance.region.astype('category').values.categories
insurance.region.astype('category').values.categories[1]
region_cat = insurance.region.astype('category').values.categories
np.random.choice(region_cat), np.random.choice(region_cat), np.random.choice(region_cat)
