from sklearn import linear_model
import numpy as np
import csv

def read_data(filename):

	reader = csv.reader(open(filename, "r"), delimiter = "\t")
	data_list= list(reader)
	data = np.array(data_list, dtype = np.float)
	
	return data

a = read_data("filtered_data.csv")
print(a)
# linear = linear_model.LinearRegression()
# linear.fit(, np.array(a[1], dtype = np.float))

# print(linear.get_params())
