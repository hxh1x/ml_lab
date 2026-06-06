#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection importtrain_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#Reading dataset
df=pd.read_csv('Boston.csv')
df.head(3)
#spliting input and output
X = df.drop(columns = ['medv'],axis=1)
y=df['medv']
#Train-testsplit
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape) 
print(X_test.shape) 
print(y_test.shape) 
#Performing simple linear regression
model = LinearRegression()
#Fitting model
model.fit(X_train, y_train)
#Prediction
y_pred = model.predict(X_test)
#Scatter plot for actual Vs predicted datapoints
sns.regplot(x=y_test, y=y_pred,scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
#Calculating error rate via performance metrics
print('Root Mean Squared error:(RMSE)',np.sqrt(mean_squared_error(Y_test,y_pred)))
print('R2-Square:',r2_score(Y_test,y_pred))
