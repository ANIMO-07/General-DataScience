# Data Visualisation and Statistics


## The Data

The file landslide_data3.csv contains a huge amount of data collected from the various landslide monitoring stations set up 
by IIT Mandi all across Himachal Pradesh, India. 
The format of the collected data is 
(dates,stationid,temperature,humidity,pressure,rain,lightavgw/o0,lightmax,moisture)


## Detailed Overview

{Question 1} I aim to study and do basic Statistical studies of all the available data in to get an idea of information 
available at hand. A basic calculation of {Mean, Median, Mode, Minimum, Maximum, Standard Deviation} were done.

{Question 2} I then also went on to create a scatter plot between rain and all other attributes and later between 
temperature and all other attributes in order to check if any correlation between the attributes could be visually recognised.

{Question 3} Then this visual observation was also corroborated by mathematically calculating the *Pearson Correlation 
Coefficient* between the aforementioned pairs of attributes.

{Question 4} A histogram for the attributes Rain and Moisture was then plotted to observe distribution over the extended period
of time over which the data had been provided for.

{Question 5} Individual histograms for rain as being grouped by stationid were also plotted.

{Question 6} Boxplot
A plot that gives the 5 point summary (Sample Min, First Quartile, Median, Third Quartile, Sample Max) of any attribute. It also
helps us recognise outliers and marks them as a hollow circle. The outliers are recognised as values that lie outside of the 
Quartile +- 1.5 IQR (Inter Quartile Range).
Under Question 6, I also made boxplots, a completely new concept for me, for the attributes Rain and Moisture and got deep
insights into how Boxplots work.


## Requirements

[numpy](https://numpy.org/)
[pandas](https://pandas.pydata.org/)
[matplotlib](https://matplotlib.org/)