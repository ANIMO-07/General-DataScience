# A. Data classification using Bayes Classifier with Gaussian Mixture Model (GMM)
# B. Regression using Simple Linear Regression and Polynomial Curve Fitting 


## The Data

This activity requires two separate datasets for the different parts within it.

A. For use in part A, we've got "seismic-bumps-train.csv" and "seismic-bumps-test.csv". These are the exact same files that
   were created as a part of Sub-project 4 from the original file "seismic_bumps1.csv". The dataset was originally collected 
   from two of longwalls located in a Polish coal mine and describes the problem of high energy (higher than 104 J) seismic 
   bumps forecasting in a coal mine. Detailed description can be found in the readme.md file of sub-project 4.

B. For use in part B, we've got "atmosphere_data.csv". The file contains the readings from various sensors installed at 10 
   locations around Mandi district. These sensors measure the different atmospheric factors like temperature, humidity, 
   atmospheric pressure, amount of rain, average light, maximum light and moisture content. The goal of this dataset is to 
   model the atmospheric temperature. 


## Detailed Overview

### Part A
This time we need to build a Bayes classifier with multimodal Gaussian Distribution with number of modes (components), Q being 
set to 2, 4, 8 and 16. 

In Bayes classifier with Multimodal Gaussian Distribution a single class is represented by the superpostition of Q number of 
different Normal Distributions also called a Gaussian Mixture Model (GMM). Hence, every class will be represented using Q means, a 
Q x Q size covariance matrix and mixture coefficients, Wq. Wq is computed as:

Wq = Number of data points in cluster q / Number of data points in class Ci

The final likelihood for any given class is calculated to be the weighted mean (weighted using the mixture coefficients) of the 
likelihoods obtained by fitting a test case into the q different curves representing the specific class. The rest of the 
procedure remains the same.

This time, we'll classify directly using the sklearn function "GaussianMixture". This will be performed and the confusion matrix
and accuracy score computed for all the different values of q.

Finally we'll tabulate and compare the best result of KNN classifier, best result of KNN classifier on normalised data, result 
of Bayes classifier using unimodal Gaussian density (all from sub-project 4) and Bayes classifier using GMM.

### Part B
Part B involves trying to predict temperature based on pressure information. For this we'll use Linear Regression which is a 
method that involves trying to plot a line over the distribution of data. The plotted line then becomes our reference and for 
given value of x-coordinate, the y-coordinate can be found and vice-versa.

Here again, our first step would be to split training data from test using the scikit-learn command "train_test_split" with 
random_state set to 42 and 70% of the data going to train and the rest to test.

#### Question 1 - Simple Linear Regression
As a part of question 1, we try to fit a straight line onto the distribution in order to try and model it. This is done using 
the inbuilt scikit-learn function "LinearRegression". The line obtained is plotted and its prediction accuracy (rmse) on both the 
training data and test data is calculated. We also plot the scatter plot of actual temperature vs predicted temperature on the 
train data.

#### Question 2 - Simple Non-linear Regression
Simple nonlinear regression involves trying to fit a polynomial of degree p on the data. As a part of question 2, we try and vary
p to find the best fit curve. Simple nonlinear regression is done by using a combination of two inbuilt functions is vscode namely,
"Polynomial Features" and "LinearRegression". Models with p = 2, 3, 4 and 5 are tested.

The prediction accuracy (rmse) for the different values of p on the training data and test data is calculated, compared and plot.
The model with lowest error (here, p = 5) is then chosen as the best fit curve and plot. We also create a scatter plot of actual
temperature vs predicted temperature on the train data using the best fit curve and the results of being more accurate are very 
clear.


## Requirements

[numpy](https://numpy.org/)  
[pandas](https://pandas.pydata.org/)  
[matplotlib](https://matplotlib.org/)  
[scikit-learn](https://scikit-learn.org/)  