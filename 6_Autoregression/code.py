'''
Name - Anmol Bishnoi
Roll Number - B19069
Branch - CSE
Mobile Number - 7042845211
'''
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv("datasetA6_HP.csv")
p = df["HP"]


#Q1

#a
df.plot()
plt.xlabel("Index of the day")
plt.ylabel("Power Consumed")
plt.show()

#b
print("Autocorrelation Coefficient =", p.autocorr())

#c
plt.scatter(p[:-1], p[1:])
plt.xlabel("Predicted Values")
plt.ylabel("Original Values")
plt.show()

#d
lag = range(1,8)
corr = []
for i in lag:
    corr.append(p.autocorr(lag=i))
plt.plot(lag, corr)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation Coefficient")
plt.show

#e
plot_acf(p, lags=7)
plt.xlabel("Lag")
plt.show()


#Q2
train, test = p[1:len(p)-250], p[len(p)-250:]
print("RMSE with persistence algorithm: \n", mse(test[1:], test[:-1], squared = False))


#Q3

#a
model = AutoReg(train, lags = 5)
model = model.fit()
predict = model.predict(start = len(train), end = len(train) + len(test))
rmse5 = mse(test, predict[:-1], squared = False)
print("RMSE =", rmse5)
plt.scatter(predict[:-1], test)
plt.xlabel("Predicted Values")
plt.ylabel("Original Values")
plt.show()

#b
lags = [1, 5, 10, 15, 25]
rmse = []
for i in lags:
    model = AutoReg(train, lags = i)
    model = model.fit()
    predict = model.predict(start=len(train), end=len(train)+len(test))
    rmse.append(mse(test, predict[:-1], squared = False))
print("Lags\t RMSE")
for i in range(5):
    print(lags[i], '\t', rmse[i])

#c
for i in range(len(lag)):
    if corr[i] > 2 / len(test)**0.5:
        hval = lag[i]
print("Heuristic Value =", hval)
print("RMSE (at vals = 5):", rmse5)