#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection importtrain_test_split
#Reading the dataset
df=pd.read_csv(‘train.csv')
df.head(3)

#-----------------------------------
#HANDLING MISSING VALUES
df.isnull().sum()
#-----------------------------------

#Dropping the “Cabin” column as it contains more null values
df = df.drop(columns='Cabin', axis=1)
#Replacing the missing values in the “Age” column with the mean value
df['Age'].fillna(df['Age'].mean(), inplace=True)
#Replacing the missing values in the “Age” column with the mode value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.isnull().sum().sum()

#----------------------------------------------
#ENCODING CATEGORICAL FEATURE
df.info()
#---------------------------------------------

#FEATURE SCALING
#spliting input and output
X = df.drop(columns = ['Survived'],axis=1)
y=df['Survived']
X.head()
#-----------------------------------------------
#Train-testsplit
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
Using Standarscalar to scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#Displaying scaled data as dataframes
scaled_df = pd.DataFrame(X_train, columns=X.columns)
scaled_df.head()
