# Attribute Normalisation, Standardisation and Dimension Reduction of Data


## The Data

This activity requires two separate datasets for the different parts within it.

1. For use in Question 1, we've got "landslide_data3.csv". The exact same dataset as used in the Sub-project 1, a huge amount 
   of data collected from the various landslide monitoring stations set up by IIT Mandi all across Himachal Pradesh, India. 
   The format of the collected data is 
   (dates,stationid,temperature,humidity,pressure,rain,lightavgw/o0,lightmax,moisture)

2. For use in Questions 2 and 3, a synthetic 2-dimensional dataset of 1000 samples is being manually generated. Each sample is
   independently and identically distributed with bi-variate Gaussian distribution with user entered mean values, μ = [0, 0]T
   and covariance matrix, Σ = [[5, 10], [10, 13]].


## Detailed Overview

### Question 1
Normalisation/Standardisation is a standard practice in Data Science and is carried out whenever we need to do major calculations
with huge amounts of data. This is mainly done because attributes in astronomical ranges turns computations done on them a highly
energy and time intensive process. Non-standardisation might also cause a skew of results in favour of attributes with bigger
values and totally undermine attributes with smaller ranges.

{part 0} Drop attributes: dates, stationid. Identify outliers in the rest of the attributes and replace them with respective medians

{part a} Try min-max normalisation on all attributes and scale them to the range 3 to 9.

{part b} Find, means and standard deviations of the original dataset and perform standardisation on the same. 

### Question 2
A multi-dimensional data might have several redundant dimensions or dimensions unrelated to the study in question. This is where the 
concept of Eigenvectors and Eigenvalues come in as they allow us to study a mutli-dimensional data source and then possibly reduce it
to a set of most important dimensions (principal components).
To learn exactly how to do this in practice, as a part of question 2, we create a synthetic 2-dimensional data of 1000 samples with 
the properties as mentioned earlier.

{part a} Plot a Scatter Plot for the same.

{part b} Compute Eigenvectors and superimpose them on the plot created earlier. This gives us a better picture of how eigenvectors work.

{part c} Project data on the two Eigenvectors separately and then plot them. This is another procedure that a Data Scientist will
need to perform frequently.

{part d} Reconstruct the data only using Principal Components and find MSE. Since dimension reduction wasn't performed, MSE = 0.

### Question 3
Now we perform the procedures tried out as a part of Question 2 on the real data from the landslide monitoring stations.
First, we perform Principal Component Analysis (PCA) on the outlier corrected standardised data as created as a part of Question 1b.
Principal Component Analysis means finding the more important dimensions which are usually created in the space by combining 
components of the other real world components and this can be showed to be the eigenvectors of the data's covariance matrix.

{part a} Reduce the 7-dimensional data to 2 dimensions. Plot the scatter plot and compare the variance along eigenvectors with
eigenvalues.

{part b} Show all Eigenvalues in descending order.

{part c} Find and show the RMSE corresponding to all the different levels of dimension reduction.


## Requirements

[numpy](https://numpy.org/)
[pandas](https://pandas.pydata.org/)
[matplotlib](https://matplotlib.org/)
[scikit-learn](https://scikit-learn.org/)