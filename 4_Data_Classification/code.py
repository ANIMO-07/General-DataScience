'''
Name - Anmol Bishnoi
Roll Number - B19069
Branch - CSE
Mobile Number - 7042845211
'''
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("seismic_bumps1.csv")          #reading the file
df = df.drop(columns = df.keys()[8:16])         #getting rid of the nbumps' columns since they are not required


#Question 1

x = df[df.columns[:-1]]
y = df[df.columns[-1]]                          #splitting parameters to x and target variable to y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, shuffle = True)
#train-test split

train = pd.concat([x_train, y_train], axis = 1)
test = pd.concat([x_test, y_test], axis = 1)    #recombining parameters and target variable to form a single dataframe

train.to_csv("seismic-bumps-train.csv")
test.to_csv("seismic-bumps-test.csv")           #saving the files as csv

#part a & b
for i in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors = i)             #creating the classifier for a given value of k
    knn.fit(x_train, y_train)                               #training the classifier
    predict = knn.predict(x_test)                           #testing the classifier
    print("\nFor k = ", i)
    print("Confusion Matrix => \n", confusion_matrix(y_test, predict))      #confusion matrix
    print("Accuracy Score => ", accuracy_score(y_test, predict))            #accuracy score


#Question 2

min = x_train.min()
max = x_train.max()
diff = max - min
normal_xtrain = (x_train - min)/diff
normal_xtest = (x_test - min)/diff              #min-max normalisation on the parameters

normal_train = pd.concat([normal_xtrain, y_train], axis = 1)
normal_test = pd.concat([normal_xtest, y_test], axis = 1)   #recombining parameters and target variable to form a single dataframe

normal_train.to_csv("seismic-bumps-train-Normalised.csv")
normal_test.to_csv("seismic-bumps-test-Normalised.csv")     #saving the files as csv

#part a & b
for i in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors = i)             #creating the classifier for a given value of k
    knn.fit(x_train, y_train)                               #training the classifier
    predict = knn.predict(x_test)                           #testing the classifier
    print("\nFor k = ", i)
    print("Confusion Matrix => \n", confusion_matrix(y_test, predict))      #confusion matrix
    print("Accuracy Score => ", accuracy_score(y_test, predict))            #accuracy score


#Question 3

train0 = train[train["class"] == 0]
train1 = train[train["class"] == 1]                         #splitting the train data based on the target variables
xtrain0 = train0[train0.columns[:-1]]
xtrain1 = train1[train1.columns[:-1]]                       #separating parameters from teh target variables

cov0 = np.cov(xtrain0.T)
cov1 = np.cov(xtrain1.T)                                    #covariance matrices for the train subsets

mean0 = np.mean(xtrain0)
mean1 = np.mean(xtrain1)                                    #means for the train subsets

def likelihood(x, m, cov):                                  #likelihood function based on the bayes model
    ex = np.exp(-0.5*np.dot(np.dot((x-m).T, np.linalg.inv(cov)), (x-m)))
    return(ex/((2*np.pi)**5 * (np.linalg.det(cov))**0.5))

prior0 = len(train0)/len(train)
prior1 = len(train1)/len(train)

predict = []
for i, x in x_test.iterrows():                              #classifying based on maximum likelihood
    p0 = likelihood(x, mean0, cov0) * prior0
    p1 = likelihood(x, mean1, cov1) * prior1
    if p0 > p1:
        predict.append(0)
    else:
        predict.append(1)

print("\nFor Bayes classifier")
print("Confusion Matrix => \n", confusion_matrix(y_test, predict))          #confusion matrix
print("Accuracy Score => ", accuracy_score(y_test, predict))                #accuracy score