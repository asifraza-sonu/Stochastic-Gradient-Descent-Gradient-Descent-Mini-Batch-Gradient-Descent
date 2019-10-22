import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv('father_son_heights.csv')
from sklearn.preprocessing import StandardScaler
X = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(x))
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
plt.scatter(X,y, color = 'blue')
plt.show()
plt.close()
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter= 1000,tol=0.0001,eta0= 1e-1) 
sgd.fit(X_train,y_train.ravel())
predictions = sgd.predict(X)
R2_Score = sgd.score(X,y)
MAE = metrics.mean_absolute_error(y,predictions)
MSE = metrics.mean_squared_error(y,predictions)
RMSE = np.sqrt(MSE)
N = len(y)
RSE = np.sqrt((N* MSE) /(N-2))
print('Predicted intercept value is: ', sgd.intercept_)
print('Predicted  coeffient value is: ',sgd.coef_ )
print('Mean absolute error (MAE) is :',MAE)
print('Mean squared error (MSE)) is :',MSE)
print('Root Mean squared error(RMSE) is :',RMSE)
print('RSE - Residual Squared Error is :',RSE)
print ('metrics_R2Score is :',metrics.r2_score(y,predictions))
print('R2 score is :',R2_Score)
plt.scatter(X,y,color = 'blue')
plt.plot(X,predictions, c='r')
fig2 = plt.gcf()
fig2.savefig('sgd_line.png')
plt.show()
plt.close()