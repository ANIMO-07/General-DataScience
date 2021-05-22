import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("landslide_data3.csv")         #Reading the file


#Question 1

q1 = {"Attributes" : ['Temperature', 'Humidity', 'Pressure', 'Rain', 'Average Light', 'Maximum Light', 'Moisture'], 
       "Mean" : [], "Median" : [], "Mode" : [], "Minimum" : [], "Maximum" : [], "Standard Deviation" : []}
#Creating a new DataFrame to store the answers to Question 1 to later print them

for keys in df.keys()[2:]:
    values = df[keys]
    q1["Mean"].append(values.mean())
    q1["Median"].append(values.median())
    q1["Mode"].append(values.mode().to_list())
    q1["Minimum"].append(min(values))
    q1["Maximum"].append(max(values))
    q1["Standard Deviation"].append(np.std(values))

q1 = pd.DataFrame(q1)
print(q1.to_string(), end="\n\n")


#Question 2

parts = ['rain', 'temperature']                                   #parts 2a and 2b
for part in parts:
    xax = df[part]
    for key in df.keys()[2:]:
        if key != part:
            yax = df[key]
            plt.scatter(xax, yax)
            plt.title('scatter plot between ' + part + ' and ' + key)
            plt.xlabel(part)
            plt.ylabel(key)
            plt.show()


#Question 3

print("Correlation Coefficents with Rain: ")
print(df.corrwith(df['rain']))
print()
print("Correlation Coefficents with Temperature: ")
print(df.corrwith(df['temperature']))


#Question 4

for part in ['rain', 'moisture']:
    hist = df.hist(column=part, bins=100)
    plt.title('Histogram of ' + part)
    plt.xlabel(part)
    plt.show()


#Question 5

for stationid, q5 in df.groupby('stationid'):                    #creating and accessing q5, a new DataFrame with everything
    hist = q5.hist(column=['rain'], bins=100)                    #grouped by their respective station ids
    plt.title('Histogram of rain at ' + stationid)
    plt.xlabel('Rain')
    plt.ylabel(stationid)
    plt.show()


#Question 6

for part in ['rain', 'moisture']:
    box = df.boxplot(column=[part])
    plt.title('Boxplot of ' + part)
    plt.show()