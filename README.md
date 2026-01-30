# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.

2.Scale the features using a standard scaler to normalize the data.

3.Initialize model parameters (theta) and add an intercept term to the feature set.

4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.

5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PANDEESWARAN N
RegisterNumber:  212224230191
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:

## Data Information : 

<img width="645" height="229" alt="image" src="https://github.com/user-attachments/assets/041a3362-e0fe-4703-bc30-9eee49919dcf" />

## Value of X : 

<img width="297" height="785" alt="image" src="https://github.com/user-attachments/assets/aba20709-c3c3-4e94-b993-0fc21aa04469" />

## Value of X1_scaled : 

<img width="374" height="782" alt="image" src="https://github.com/user-attachments/assets/6d4f7b22-07fc-4ef6-a8ae-df1b8ea02c7f" />

## predicted value :

<img width="403" height="55" alt="image" src="https://github.com/user-attachments/assets/c7a070ac-68f0-4a7f-a4e8-12eeaa619472" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
