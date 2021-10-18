# Autoregression


## The Data

The dataset used in this project is MNIST dataset. The MNIST dataset is a widely used image database. It comprises a 
large number of images of handwritten digits from 0 to 9. Each image is of size 28 x 28. For this sub-project, instead
of using the MNIST dataset directly, the images are instead vectorized to a dimension of 784. The 784-dimensional vectors 
are then reduced to 2-dimensional vectors using t-Distributed Stochastic Neighbour Embedding (t-SNE) dimensionality 
reduction technique. The details of the t-SNE technique can be found [here](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1).

The data is contained in the files "mnist-tsne-train.csv" and "mnist-tsne-test.csv" files. The train file contains 1000
tuples of the 2-dimensional data which includes 100 examples for each of the 10 classes. Third column is the class 
information. Similarly mnist-tsne-test.csv contains 500 tuples of the 2-dimensional data which includes 50 examples for 
each of the 10 classes.


## Detailed Overview

The main task here is to partition/cluster the training data into 10 clusters using different clustering techniques.


## Requirements

[numpy](https://numpy.org/)  
[pandas](https://pandas.pydata.org/)  
[matplotlib](https://matplotlib.org/)  
[scikit-learn](https://scikit-learn.org/)  
[scipy](https://www.scipy.org/)  