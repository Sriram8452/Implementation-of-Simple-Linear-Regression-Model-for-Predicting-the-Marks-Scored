![4](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/5c7dbc15-479f-42d9-bcb7-a8e4f0e64d2e)# Ex.No.2-Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sriram G
RegisterNumber:212222230149
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

![1](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/972b4cb3-6c85-4e10-8672-91b5cc66a7c9)

![2](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/81b12262-55a4-4835-b655-920313aadb5f)

![3](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/b6e03dc6-b1ae-4eb9-b39a-d292e670ec2a)

![4](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/68cc3217-1f57-42ab-8ffb-0aa0d3e896b2)

![5](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/53651595-7c2c-46bb-8373-63088d2c9221)

![6](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/815c37f2-c9af-4e40-ba49-8c8ed4ef8fc9)

![7](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/46c655b5-32ef-4e48-abaf-b792b26aeb54)

![8](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/88deac0f-8244-409a-bdca-66f047759537)

![9](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/30a5d49f-1bcc-47ed-9e51-1c02214a6993)

![10](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/a950bd0d-7852-4e84-8038-11be30b6bffa)

![11](https://github.com/Sriram8452/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708032/5e1fe719-68e7-403e-bf64-4d19d37dcdba)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
