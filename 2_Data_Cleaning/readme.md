# Data Cleaning
Handling Missing Values and Outlier Analysis


## The Data

The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective was 
to predict based on diagnostic measurements whether a patient has diabetes. The downloaded data is in the file 
"pima_indians_diabetes_original.csv". A few values have been randomly removed from the dataset and the resulting 
data was saved in the file "pima_indians_diabetes_miss.csv".
The format of the data is (pregs,plas,pres,skin,test,BMI,pedi,Age,class)


## Detailed Overview

{Question 1} A simple plot to find the number of missing values in each attribute.

{Question 2a} Rejecting useless tuples - ones with one third of their values missing.
{Question 2b} Rejecting useless tuples - ones with target class missing.

{Question 3} Counting the number of missing values again.

{Question 4} Replacing the current missing values first by the attribute Mean and then using the Linear Interpolation 
using values just above and just below the missing value (inbuilt functions - df.mean() and df.interpolate()).

{Question 4_a} Then the values of the measures of Central Tendency viz. Mean, Median, Mode, Standard Deviation were 
calculated forthe original data, mean substituted data and the interpolated data and then compared.

{Question 4_b} Lastly, the Root Mean Square Error, in estimating the data by using our two methods, when compared with 
the true values, as obtained from the original file, for every individual attribute was calculated. We quickly learnt 
that mean substitution is inaccurate while interpolation was equally worse with the non-sequential nature of the data we had.

{Question 5} Next, the boxplots for the attributes Age and BMI were created and the outliers obtained for the same. These
outliers were then replaced by the sample medians and new boxplots were created. Formation of new outliers observed.


## Requirements

[numpy](https://numpy.org/)  
[pandas](https://pandas.pydata.org/)  
[matplotlib](https://matplotlib.org/)  