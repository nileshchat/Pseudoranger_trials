'''
Date Created: 24th March, 2017
Author: Nilesh Chaturvedi

Algorithm:
Equation of an ellipse : x^2/a^2 + y^2/b^2 = 1
						 y^2/a^2 = 1 - x^2/b^2
						 y^2 = a^2 - (a^2/b^2)*x^2
						Let y^2 : Y
						 	x^2 : X
						 	a^2 : intercept
						 	a^2/b^2 : w1
So the regression model : Y = intercept - w1*X						
	
'''

from sklearn import linear_model
import numpy as np
import csv
import math

def read_data(filename):

	reader = csv.reader(open(filename, "r"), delimiter = ",")
	data_list= list(reader)
	data = np.array(data_list, dtype = np.float)
	
	return data

filedata = read_data("filtered_data.csv")
X = []
Y = []
for row in filedata:
	X.append(row[0])
	Y.append(row[1])
x_coord = [[i] for i in X]
y_coord = [[i] for i in Y]

curve = linear_model.LinearRegression()
curve.fit(x_coord, y_coord)



