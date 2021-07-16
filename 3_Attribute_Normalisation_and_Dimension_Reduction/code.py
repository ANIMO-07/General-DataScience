import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

df = pd.read_csv("landslide_data3.csv")         #Reading the file


#Question 1

df = df.drop(columns=['dates', 'stationid'])

for key in df.keys():
    vals = []
    _, box = df.boxplot(column=[key], return_type='both')                       #Making boxplot to fetch outliers
    outliers = [flier.get_ydata() for flier in box["fliers"]]
    for i, record in df.iterrows():
        if record[key] in outliers[0]:
            vals.append(i)
    dfd = df.drop(vals)
    for val in vals:
        df.loc[val , key] = dfd[key].median()                                   #Replacing outliers with medians

plt.clf()

#Question 1a

min = df.min()
max = df.max()

print("Before Normalisation")
print('Dictionary of Minimum Values')
print(min)
print('Dictionary of Maximum Values')
print(max)

diff = max - min
df1a = (df - min)/diff * 6 + 3

min1a = df1a.min()
max1a = df1a.max()

print("After Normalisation")
print('Dictionary of Minimum Values')
print(min1a)
print('Dictionary of Maximum Values')
print(max1a)


#Question 1b

avg = df.mean()
sd = df.std()

print("Before Standardisation")
print('Mean')
print(avg)
print('Standard Deviation')
print(sd)

df1b = (df - avg)/sd

avg1b = df1b.mean()
sd1b = df1b.std()

print("After Standardisation")
print('Mean')
print(avg1b)
print('Standard Deviation')
print(sd1b)


#Question 2

covar = [[5, 10], [10, 13]]
D = np.random.multivariate_normal(mean = [0,0], cov = covar, size = 1000, check_valid = 'ignore')
D = pd.DataFrame(D, columns=['A', 'B'])
# Creating the 2-dimensional synthetic data with mean, μ = [0, 0]T and covariance matrix, Σ = [[5, 10], [10, 13]].


#Question 2a
plt.scatter(D['A'], D['B'])
plt.show()


#Question 2b
cov_matrix = np.dot(np.transpose(D), D)/1000
val, vec = np.linalg.eig(cov_matrix)            #This gives Eigenvectors and Eigenvalues from the Covariance Matrix
print('The eigenvalues are:')
print(val)
print('The eigenvectors are:')
print(vec)

plt.scatter(D['A'], D['B'])
plt.quiver(0, 0, vec[0][0], vec[1][0], scale=5, angles = 'xy')
plt.quiver(0, 0, vec[0][1], vec[1][1], scale=2, angles = 'xy')
plt.show()


#Question 2c

A = np.dot(D, vec)

for i in range(2):
    xx = []
    yy = []
    for d in A:
        xx.append(d[i]*vec[0][i])                       #x-coordinate of data projected along the Ith Eigenvector
        yy.append(d[i]*vec[1][i])                       #y-coordinate of data projected along the Ith Eigenvector
    plt.scatter(D['A'], D['B'])
    plt.scatter(xx, yy)
    plt.quiver(0, 0, vec[0][0], vec[1][0], scale=5, angles = 'xy')
    plt.quiver(0, 0, vec[0][1], vec[1][1], scale=2, angles = 'xy')
    plt.show()


#Question 2d

pca = decomposition.PCA(n_components = 2)
proj = pca.fit_transform(D)
#This fucntion helps us directly find the data as projected along the first n-most significant eigenvectors where
#n is already specified as a part of the previous statement as n_components.
recon = pca.inverse_transform(proj)
#This function reconstructs the data along original dimensions from the projected data using information about the
#eigenvectors and their directions as a function of the original components.
sum = ((D - recon)**2).sum()
rmse = (sum/(len(D)*2))**0.5
mse = rmse.sum()
print('MSE =', mse)


#Question 3

pca = decomposition.PCA(n_components = 2)


#Question 3a

df3a = pca.fit_transform(df1b)
df3a = pd.DataFrame(df3a, columns=['A', 'B'])           #Projected data along the first 2 most significant eigenvectors

cov_matrix = np.dot(np.transpose(df1b), df1b)/len(df1b)
val, vec = np.linalg.eig(cov_matrix)

print("Largest 2 eigenvalues are")
print(val[:2])                                          #2 most significant eigenvalues
print("Variance is")
print(df3a.var())                                       #Variance in Projected Data
plt.scatter(df3a['A'], df3a['B'])
plt.show()


#Question 3b

val = sorted(list(val), reverse = True)                 #Eigenvalues ranked in significance
x = [1, 2, 3, 4, 5, 6, 7, 8]
plt.bar(list(range(len(val))), val)
plt.title("Eigenvalues")
plt.show()


#Question 3c

errs = []
for i in range(1, 8):
    pca = decomposition.PCA(n_components = i)
    proj = pca.fit_transform(df1b)                      #Projected Data along first i eigenvectors
    recon = pca.inverse_transform(proj)                 #Reconstructed back to the original 7 dimensions
    sum = ((df1b - recon)**2).sum()
    rmse = (sum/(len(df1b)*7))**0.5
    rmse = rmse.sum()
    errs.append(rmse)

l = [1, 2, 3, 4, 5, 6, 7]
plt.plot(l, errs)
plt.xlabel('Values of l')
plt.ylabel('RMSE Errors')