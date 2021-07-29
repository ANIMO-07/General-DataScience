# Data Classification using KNN and Bayes Classifier with Unimodal Gaussian Density


## The Data

The dataset Seismic-Bumps Data Set describes the problem of high energy (higher than 104 J) seismic bumps forecasting in a
coal mine. This data is collected from two of longwalls located in a Polish coal mine. The data is saved as the file
"seismic_bumps1.csv"

The other files, namely "seismic-bumps-train.csv", "seismic-bumps-test.csv", "seismic-bumps-train-Normalised.csv" and
"seismic-bumps-test-Normalised.csv" shall get created later on as a part of the different questions under the sub-project.

### Additional Information
Mining activity was and is always connected with the occurrence of dangers which are commonly called mining hazards. A
special case of such threat is a seismic hazard which frequently occurs in many underground mines. Seismic hazard is the 
hardest detectable and predictable of natural hazards and in this respect it is comparable to an earthquake. More and more 
advanced seismic and seismoacoustic monitoring systems allow a better understanding rock mass processes and definition of 
seismic hazard prediction methods. Accuracy of so far created methods is however far from perfect. Complexity of seismic 
processes and big disproportion between the number of low-energy seismic events and the number of high-energy phenomena 
(e.g. > 104 J) causes the statistical techniques to be insufficient to predict seismic hazard. 

This dataset contains recorded features from the seismic activity in the rock mass and seismoacoustic activity with the 
possibility of rockburst occurrence to predict the hazardous and non-hazardous state. It consists 2584 tuples each having 19
attributes. The last attribute for every tuple signifies the class label (0 for hazardous state and 1 for non-hazardous state). 
It is a two class problem. Other attributes are input features.

### Attribute Information
1. *seismic*:        result of shift seismic hazard assessment in the mine working obtained by the seismic method 
    (1 - lack of hazard, 2 - low hazard, 3 - high hazard, 4 - danger state)
2. *seismoacoustic*: result of shift seismic hazard assessment in the mine working obtainedby the seismoacoustic method 
    (1 - lack of hazard, 2 - low hazard, 3 - high hazard, 4 - danger state)
3. *shift*:          information about type of a shift (W - coal-getting, N -preparation shift)
4. *genergy*:        seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones 
    monitoring the longwall
5. *gpuls*:          a number of pulses recorded within previous shift by Gmax
6. *gdenergy*:       a deviation of energy recorded within previous shift by GMax from average energy recorded during eight 
    previous shifts
7. *gdpuls*:         a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses 
    recorded during eight previous shifts
8. *ghazard*:        result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based 
    on registration coming from GMax only
9. *nbumps*:         the number of seismic bumps recorded within previous shift
10. *nbumps2*:       the number of seismic bumps (in energy range [10^2, 10^3)) registered within previous shift
11. *nbumps3*:       the number of seismic bumps (in energy range [10^3, 10^4)) registered within previous shift
12. *nbumps4*:       the number of seismic bumps (in energy range [10^4, 10^5)) registered within previous shift
13. *nbumps5*:       the number of seismic bumps (in energy range [10^5, 10^6)) registered within the last shift
14. *nbumps6*:       the number of seismic bumps (in energy range [10^6, 10^7)) registered within previous shift
15. *nbumps7*:       the number of seismic bumps (in energy range [10^7, 10^8)) registered within previous shift
16. *nbumps89*:      the number of seismic bumps (in energy range [10^8, 10^10)) registered within previous shift
17. *energy*:        total energy of seismic bumps registered within previous shift;
18. *maxenergy*:     the maximum energy of the seismic bumps registered within previous shift;
19. *class*:         the decision attribute - '1' means that high energy seismic bump occurred in the next shift ('hazardous state'),
    '0' means that no high energy seismic bumps occurred in the next shift ('non-hazardous state')


## Detailed Overview

### Question 1
First, the dataset was split into train and test using the inbuilt sklearn function, "train_test_split", with 70% of the data being
a part of train and the rest going to test. The parameter, "random_state" for the function was set at 42. This split data was 
subsequently saved into the files "seismic-bumps-train.csv" and "seismic-bumps-test.csv" respectively.

Then classification based on the K-Nearest Neighbour (KNN) method was tested for the different values of K(1, 3 & 5). KNN is a 
popular classifying method which classifies any given test case based on the classes of the K Nearest Neigbours (odd to prevent 
conflict) to it.
KNN was performed using the inbuilt sklearn function "KNeighborsClassifier" and the accuracy of the classifier was subsequently 
tested by calculating the Confusion Matrix and the Accuracy Score.

### Question 2
KNN uses simple euclidean distance to calculate K Nearest Neighbours. This can sometimes pose a problem when the ranges of the 
components is particularly diverse where one component ranges from 0.1unit to 0.3unit where the other ranges from 1M to 100M.
This problem is particularly tackled by normalising the entire dataset.

As a part of Q2, that is exactly what is done i.e. KNN on normalised data. We first normalise both our train and test using
min-max normalisation. Here, the min and max are obtained from the training set. Then we perform KNN again using the values of K at
1, 3 and 5 and then calculating both the confusion matrix and the accuracy score for each.

### Question 3
As a part of q3, we perform classification here using yet another method, namely the Naive Bayes Classifier. This method requires
us to build Normal Distribution/bell curve/gaussian curve for the various target variables. The classifiers need to be built 
beforehand and hence require preprocessing unlike KNN which requires zero preprocessing. On the other hand KNN needs a lot of 
computation in the testing phase which is not true for Bayes.

After building the models individual test cases are put into the model and the **likelihood** of the testcase being a part of the 
various different classes is calculated. The formula for likelihood is as follows:

![Likelihood Formula](./Likelihood.png?raw=true)

Here, X is the testcase
      Σ is the covariance matrix
      μ is the mean

After calculating the likelihood for individual classes is calculated, they're multiplied by **prior**, a tool used to take care of
discrepancies based on unequal number of train samples for the different classes. The formula for prior is:

Prior = Number of train tuples of class x/Total number of train tuples.

After multiplying by prior, the value we get is the posterior probablility. Any train data tuple is assigned the class which it has 
the maximum value of Posterior Probability for.

Yet again, we calculate the Confusion Matrix and the Accuracy Score for this method

### Question 4
As a part of Q4, we simply tabulate and compare the best results obtained from our three different methods namely the KNN classifier,
KNN classifier on Normalised Data and Bayes Classifier on Unimodal Gaussian Density.


## Requirements

[numpy](https://numpy.org/)
[pandas](https://pandas.pydata.org/)
[scikit-learn](https://scikit-learn.org/)