'''
Name - Anmol Bishnoi
Roll Number - B19069
Branch - CSE
Mobile Number - 7042845211
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Part A
#Question 1

train = pd.read_csv("seismic-bumps-train.csv")
test = pd.read_csv("seismic-bumps-test.csv")

x_train = train[train.columns[1:-1]]
y_train = train[train.columns[-1]]

train0 = train[train["class"] == 0]
train1 = train[train["class"] == 1]
xtrain0 = train0[train0.columns[1:-1]]
xtrain1 = train1[train1.columns[1:-1]]
x_test = test[test.columns[1:-1]]
y_test = test[test.columns[-1]]

for q in [2, 4, 8, 16]:
    gmm0 = GaussianMixture(n_components = q, covariance_type = "full", random_state = 42)
    gmm1 = GaussianMixture(n_components = q, covariance_type = "full", random_state = 42)
    gmm0.fit(xtrain0)
    gmm1.fit(xtrain1)
    likelihood0 = gmm0.score_samples(x_test) + np.log(len(train0)/len(train))
    likelihood1 = gmm1.score_samples(x_test) + np.log(len(train1)/len(train))
    predict = []
    for i in range(len(x_test)):
        if likelihood0[i] > likelihood1[i]:
            predict.append(0)
        else:
            predict.append(1)
    print("\nFor Q = ", q)
    print("Confusion Matrix => \n", confusion_matrix(y_test, predict))
    print("Accuracy Score => ", accuracy_score(y_test, predict))


#Part B

#Question 1

df = pd.read_csv("atmosphere_data.csv")

x = df[df.columns[:-1]]
y = df[df.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, shuffle = True)

train = pd.concat([x_train, y_train], axis = 1)
test = pd.concat([x_test, y_test], axis = 1)

line = LinearRegression()
f = np.array(x_train["pressure"]).reshape(-1,1)
line.fit(f, y_train)


#subpart a
curvex = np.linspace(400, 1100)
curvex = np.array(curvex).reshape(-1,1)
curvey = line.predict(curvex)
plt.scatter(x_train["pressure"], y_train)
plt.plot(curvex, curvey, color = "red")
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title('Best linear fit curve')
plt.show() 

#subpart b

predict_train = line.predict(f)
print("\nRMSE for Training Data Set on Linear Regression Model:")
print(mse(y_train, predict_train, squared = False))


#subpart c

ft = np.array(x_test["pressure"]).reshape(-1,1)
predict_test = line.predict(ft)
print("\nRMSE for Test Data Set on Linear Regression Model:")
print(mse(y_test, predict_test, squared = False))


#subpart d

plt.scatter(y_test, predict_test)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Linear Regression Model')
plt.show()


#Question 2

def poly(p, data):
    polynomial_features = PolynomialFeatures(degree = p)
    x_poly = polynomial_features.fit_transform(f)
    func = LinearRegression()
    func.fit(x_poly, y_train)
    data = np.array(data).reshape(-1,1)
    xt_poly = polynomial_features.fit_transform(data)
    predict_test = func.predict(xt_poly)
    return(predict_test)

#subpart a
print("\nRMSE values for Training Data Set on non-linear regression model is as follows:")
rmse = []
for p in [2, 3, 4, 5]:
    print("P =", p)
    q = (mse(y_train, poly(p, x_train["pressure"]), squared = False))
    rmse.append(q)
    print(q)
'''
plt.bar([2,3,4,5], rmse)
plt.xlabel("p value")
plt.ylabel("RMSE")
plt.show()'''


#subpart b
print("\nRMSE values for Test Data Set on non-linear regression model is as follows:")
rmse = []
for p in [2, 3, 4, 5]:
    print("P =", p)
    q = (mse(y_test, poly(p, x_test["pressure"]), squared = False))
    rmse.append(q)
    print(q)
'''
plt.bar([2,3,4,5], rmse)
plt.xlabel("p value")
plt.ylabel("RMSE")
plt.show()'''


#subpart c
curvex = np.linspace(400, 1100)
curvey = poly(5, curvex)
plt.scatter(x_train["pressure"], y_train)
plt.plot(curvex, curvey, color = "red")
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title('Best non-linear fit curve')
plt.show()


#subpart d

plt.scatter(y_test, poly(5, x_test["pressure"]))
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Non-Linear Regression Model')
plt.show()