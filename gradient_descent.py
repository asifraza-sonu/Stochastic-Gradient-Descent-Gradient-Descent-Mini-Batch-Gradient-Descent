import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('father_son_heights.csv')
dataset.columns
#X = dataset['Father'].values.reshape(-1,1)
#y = dataset['Son'].values.reshape(-1,1).ravel()
X = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
dataset.describe
plt.scatter(X,y)
plt.show()
plt.close()
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

#Function to create Mini Batches
def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) 

    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches

#Now for Gradient Descent  which is other wise called as Batch gradient descent, as suggested in the question paper we can do pass the
#Batch size as whole training data set,then it will be a implementation of a gradient descent

whole_batch_size = X.shape[0]
batches = create_mini_batches(X_train,y_train,whole_batch_size)
Model = linear_model.SGDRegressor(max_iter = 1000,tol=0.001,eta0 = 1e-1, shuffle=True)
batches_list = list(batches)
#for batch in batches :
for _ in range(5):
    np.random.shuffle(batches_list)
    for X_chunk,y_chunk in batches_list:
           Model.partial_fit(X_chunk,y_chunk.ravel())
y_predicted = Model.predict(X)
mse = metrics.mean_squared_error(y, y_predicted)
MAE = metrics.mean_absolute_error(y,y_predicted)
MSE = metrics.mean_squared_error(y,y_predicted)
RMSE = np.sqrt(MSE)
N = len(y)
RSE = np.sqrt((N* MSE) /(N-2))
print("RMSE: ",np.sqrt(mse))
print('Predicted intercept value is: ',Model.intercept_ )
print('Predicted  coeffient value is: ',Model.coef_ )
print ('R2_score is ',Model.score(X_chunk, y_chunk))
print('Mean absolute error (MAE) is :',MAE)
print('Mean squared error (MSE)) is :',MSE)
print('Root Mean squared error(RMSE) is :',RMSE)
print('RSE - Residual Squared Error is :',RSE)
plt.scatter(X,y, c='b')
plt.plot(X,y_predicted , c= 'r')
fig4 = plt.gcf()
fig4.savefig('gradient_descent_line.png')
plt.show()
plt.close()