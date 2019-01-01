#IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier

df_train = pd.read_csv("d:/downloads/datasets/mahindra iic/train_new.csv").dropna()
# Read data set into pandas dataframe and drop the rows with missing values.
y = df['Target']
# Store the Target values in y
df.drop('Target',axis=1,inplace=True)
# Separate the data set from the target value for fitting the model


#************************ PREPROCESSING TEST & TRAIN DATA SETS *******************************************************************************************
#
# Evaluating data columns from Excel sheet and looking for redundant data columns.
# In this case, data in Column9 was unique for each row. Whereas ID Column1 is already unique for each row. Hence eliminating Column9
# In case of Column15, Column17 & Column21 more than half of the values are NA or empty. Hence eliminate these columns
# In case of Column8 & Column16 the possible values are Y/N/NaN. Hence in place of NaN (null or empty) we populate N as it has a greater mode value.
# In case of Column22 the missing values can be replaced with the 1150 (highest mode value)
# Similarly, for Column23 100 has the highest mode and can be replace the missing values.
# A few rows of some Columns have missing values. This can be handled in the rest of the code.
#
#*********************************************************************************************************************************************************

df_train.reset_index(inplace=True)
# Reset Index of the dataframe, which was disturbed due to deletion of missing value rows.
# Rename the new index column as Uindex

df.drop('Column9',axis=1,inplace=True)
df.drop('Column15',axis=1,inplace=True)
df.drop('Column17',axis=1,inplace=True)
df.drop('Column21',axis=1,inplace=True)

df[Column8] = df[Column8].fillna('N')
df[Column16] = df[Column16].fillna('N')
df[Column22] = df[Column22].fillna('1150')
df[Column23] = df[Column23].fillna('100')


# Converting Categorical Values to Numeric values using Label Encoder library
# Repeating the same for all data Columns in the dataframe
le = preprocessing.LabelEncoder()

for colName in df.columns:
    le.fit(df[colName])
    df[colName] = le.transform(df[colName])


df.to_csv("d:/downloads/datasets/mahindra iic/train_new1.csv")
# Store the cleaned training data set


# ************ REPEATING THE ABOVE STEPS FOR TEST DATA SET ************************************************************8

df_test = pd.read_csv("d:/downloads/datasets/mahindra iic/test_data2.csv").dropna()

df_test.reset_index(inplace=True)
# Reset Index of the dataframe, which was disturbed due to deletion of missing value rows.
# Rename the new index column as Uindex

df_test.drop('Column9',axis=1,inplace=True)
df_test.drop('Column15',axis=1,inplace=True)
df_test.drop('Column17',axis=1,inplace=True)
df_test.drop('Column21',axis=1,inplace=True)

df_test[Column8] = df_test[Column8].fillna('N')
df_test[Column16] = df_test[Column16].fillna('N')
df_test[Column22] = df_test[Column22].fillna('1150')
df_test[Column23] = df_test[Column23].fillna('100')

for colName in df_test.columns:
    df_test[colName] = df_test[colName].fillna('0')
    le.fit(df_test[colName])
    df_test[colName] = le.transform(df_test[colName])


df_test.to_csv("d:/downloads/datasets/mahindra iic/test2_data2.csv")

# ************************** START ALGORITHM ****************************************************************************8

df = pd.read_csv("d:/downloads/datasets/mahindra iic/train_new2.csv")
# Read the cleaned Training Data Set
df_test = pd.read_csv("d:/downloads/datasets/mahindra iic/test2_data2.csv")
# Read the cleaned Testing Data Set 1


# Fitting AdaBoostClassifier Algorithm on the training data set
ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
ada.fit(df,y)

y_pred = ada.predict(df_test)
# Store the predicted values in y_pred
df_out = pd.DataFrame(y_pred, columns=['Class'])
df_out.to_csv("d:/downloads/datasets/mahindra iic/MachineLearning_Sagar_Patil.csv")
# Store the result as per submission format

# ************* Some Algorithms that weren't so accurate ***************************************
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(df,y)
# y_pred = clf.predict(df_test)


# ptn = Perceptron(max_iter=10000)
# ptn.fit(df,y)
# y_pred = ptn.predict(df_test)
# #labels = range (y_pred.shape[1])
# df_out = pd.DataFrame(y_pred, columns=['y_pred'])
# #df_out.add(df_test['ID'])
# df_out.to_csv("d:/downloads/datasets/mahindra iic/MachineLearning_Sagar_Patil.csv")
# print(df_out)
