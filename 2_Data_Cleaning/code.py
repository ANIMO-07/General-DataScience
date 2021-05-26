import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pima_indians_diabetes_miss.csv")         #Reading the file
dfo = pd.read_csv("pima_indians_diabetes_original.csv")

#Question 1

y=[]
for key in df.keys():
    y.append(df[key].isna().sum())

plt.bar(df.keys(),y)
plt.title('No. of missing values in various attributes')
plt.show()


#Question 2a

rows2a=[]
for i, record in df.iterrows():
    if record.isna().sum()>=3:
        rows2a.append(i)

df = df.drop(rows2a)
print(len(rows2a), "tuples were deleted")
print("The row numbers for the tuples deleted are", rows2a)


#Question 2b

rows2b=[]
for i, record in df.iterrows():
    if record.isna()['class'] == True:
        rows2b.append(i)

df = df.drop(rows2b)
print(len(rows2b), "tuples were deleted")
print("The row numbers for the tuples deleted are", rows2b)


#Question 3

miss = lambda key: df[key].isna().sum()
t=0
print("The number missing values in the different attributes are:")
for key in df.keys():
    n=miss(key)
    print(key, '-', n)
    t+=n
print("The total number of missing values in the file are", t)


#Question 4

original = {"Attributes" : ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 
        'Diabetes Pedigree Function', 'Age', 'Outcome'], 
       "Mean" : [], "Median" : [], "Mode" : [], "Standard Deviation" : []}

for keys in dfo.keys():
    values = dfo[keys]
    original["Mean"].append(values.mean())
    original["Median"].append(values.median())
    original["Mode"].append(values.mode().to_list())
    original["Standard Deviation"].append(np.std(values))

original = pd.DataFrame(original)
print('Original')
print(original)

dfo = dfo.drop(rows2a)
dfo = dfo.drop(rows2b)

def rmse(x):
    y=[]
    for key in x.keys():
        sum = ((x.loc[ : , key] - dfo.loc[ : , key])**2).sum()
        na = miss(key)
        if na == 0:
            ans = 0
        else:
            ans = (sum/na)**0.5
        y.append(ans)

    plt.bar(df.keys(),y)
    plt.title('RMSE for the different attributes')
    plt.show()
    

#Question 4a

dfm = df.fillna(df.mean())

with_mean = {"Attributes" : ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 
        'Diabetes Pedigree Function', 'Age', 'Outcome'], 
       "Mean" : [], "Median" : [], "Mode" : [], "Standard Deviation" : []}
#Creating a new DataFrame to store the answers to Question 1 to later print them

for keys in dfm.keys():
    values = dfm[keys]
    with_mean["Mean"].append(values.mean())
    with_mean["Median"].append(values.median())
    with_mean["Mode"].append(values.mode().to_list())
    with_mean["Standard Deviation"].append(np.std(values))

with_mean = pd.DataFrame(with_mean)
print("After substituting with mean")
print(with_mean)
rmse(dfm)


#Question 4b

dfi = df.interpolate()

with_interpolation = {"Attributes" : ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 
        'Diabetes Pedigree Function', 'Age', 'Outcome'], 
       "Mean" : [], "Median" : [], "Mode" : [], "Standard Deviation" : []}
#Creating a new DataFrame to store the answers to Question 1 to later print them

for keys in dfi.keys():
    values = dfi[keys]
    with_interpolation["Mean"].append(values.mean())
    with_interpolation["Median"].append(values.median())
    with_interpolation["Mode"].append(values.mode().to_list())
    with_interpolation["Standard Deviation"].append(np.std(values))

with_interpolation = pd.DataFrame(with_interpolation)
print("After Interpolation")
print(with_interpolation)
rmse(dfi)


#Question 5

for part in ['Age', 'BMI']:
    _, box = dfi.boxplot(column=[part], return_type='both')
    plt.title('Boxplot of ' + part + ' before outlier removal')
    outliers = [flier.get_ydata() for flier in box["fliers"]]
    print(outliers)
    plt.show()
    for i, record in dfi.iterrows():
        if record[part] in outliers[0]:
            record[part] = dfi[part].median()

for part in ['Age', 'BMI']:
    box = dfi.boxplot(column=[part])
    plt.title('Boxplot of ' + part + ' after outlier removal')
    plt.show()