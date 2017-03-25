'''
Date Created: 25th March, 2017
Author: Nilesh Chaturvedi

This code is inspired from the article named "Direct Least Squares Fitting of 
Ellipses" by Andrew W. Fitzgibbon, Maurizio Pilu and Robert B. Fisher

Homogeneous Equation: a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
Here [a,b,c,d,e,f] is the coef vector being used.

'''
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

def load_data(filename):
	reader = csv.reader(open(filename, "r"), delimiter = ",")
	data_list= list(reader)
	data = np.array(data_list, dtype = np.float)
	X = []
	Y = []	
	for row in data:
		X.append(row[0])
		Y.append(row[1])
	X = np.array(X)[:,np.newaxis]
	Y = np.array(Y)[:,np.newaxis]
	
	return X, Y

def fit(x, y):
	data_vector = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(y)))
	C_matrix = np.zeros([6,6])
	C_matrix[0,2] = C_matrix[2,0] = 2; C_matrix[1,1] = -1
	scatter_matrix = np.dot(data_vector.T, data_vector)
	E, V =  np.linalg.eig(np.dot(np.linalg.inv(scatter_matrix), C_matrix))
	max_val = np.argmax(np.abs(E))
	coefs = V[:,max_val]

	return coefs

def center(coef_vector):
	a,b,c,d,e,f = coef_vector[0], coef_vector[1]/2, coef_vector[2], coef_vector[3]/2, coef_vector[4]/2, coef_vector[5]
	delta = b*b-a*c
	center_x=(c*d-b*e)/delta
	center_y=(a*e-b*d)/delta

	return center_x, center_y

def axes(coef_vector):
	a,b,c,d,e,f = coef_vector[0], coef_vector[1]/2, coef_vector[2], coef_vector[3]/2, coef_vector[4]/2, coef_vector[5]
	det = 2*(a*e*e+c*d*d+f*b*b-2*b*d*e-a*c*f)
	semi_major = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a)) / det
	semi_minor = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a)) / det

	return semi_major, semi_minor

def tilt_angle(coef_vector):
	a,b,c,d,e,f = coef_vector[0], coef_vector[1]/2, coef_vector[2], coef_vector[3]/2, coef_vector[4]/2, coef_vector[5]
	if b == 0:
		if a > c:
			return 0
		else:
			return np.pi/2
	else:
		if a > c:
			return np.arctan(2*b/(a-c))/2
		else:
			return np.pi/2 + np.arctan(2*b/(a-c))/2

if __name__ == "__main__":
	filedata = load_data("filtered_data.csv")
	coefficients = fit(filedata[0], filedata[1])
	center_x, center_y = center(coefficients)
	semi_major, semi_minor = axes(coefficients)
	angle = tilt_angle(coefficients)
	angle_domain = np.arange(0,2*np.pi, 0.01)

	X = center_x + semi_major*np.cos(angle_domain)*np.cos(angle) - semi_minor*np.sin(angle_domain)*np.cos(angle)
	Y = center_y + semi_major*np.cos(angle_domain)*np.sin(angle) + semi_minor*np.sin(angle_domain)*np.cos(angle)

	plt.plot(X, Y, "g", label="Least Square fit")
	plt.plot(center_x, center_y, "*", label=center)
	plt.plot(filedata[0], filedata[1], "y", label = "raw data")
	plt.grid(True)
	plt.show()