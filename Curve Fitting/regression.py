'''
Date Created: 24th March, 2017
Author: Nilesh Chaturvedi
'''
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

def read_data(filename):

	reader = csv.reader(open(filename, "r"), delimiter = ",")
	data_list= list(reader)
	data = np.array(data_list, dtype = np.float)
	
	return data

def load_data(filename):

    reader = csv.reader(open(filename, "r"), delimiter = "\t")
    data_list= list(reader)
    x_comp = []
    y_comp = []

    for i in data_list[1:]:
        x_comp.append(i[4])
        y_comp.append(i[5])
    
    return np.array(x_comp, dtype = np.float), np.array(y_comp, dtype = np.float)

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    coefs = V[:,n]
    return coefs

def ellipse_center(a):
    a,b,c,d,e,f = a[0], a[1]/2, a[2], a[3]/2, a[4]/2, a[5]
    num = b*b-a*c
    x0=(c*d-b*e)/num
    y0=(a*e-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    a,b,c,d,e,f = a[0], a[1]/2, a[2], a[3]/2, a[4]/2, a[5]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    a,b,c,d,e,f = a[0], a[1]/2, a[2], a[3]/2, a[4]/2, a[5]
    up = 2*(a*e*e+c*d*d+f*b*b-2*b*d*e-a*c*f)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    a,b,c,d,e,f = a[0], a[1]/2, a[2], a[3]/2, a[4]/2, a[5]
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

filedata = load_data("track1489609280.csv")
# X = []
# Y = []
# for row in filedata:
# 	X.append(row[4])
# 	Y.append(row[5])
curve = fitEllipse(np.array(filedata[0]), np.array(filedata[1]))

arc = 2
R = np.arange(0,arc*np.pi, 0.01)

angle = ellipse_angle_of_rotation2(curve)
center = ellipse_center(curve)
axes = ellipse_axis_length(curve)

a, b = axes

xx = center[0] + a*np.cos(R)*np.cos(angle) - b*np.sin(R)*np.sin(angle)
yy = center[1] + a*np.cos(R)*np.sin(angle) + b*np.sin(R)*np.cos(angle)

print(len(xx))

plt.plot(xx, yy)
plt.plot(filedata[0], filedata[1], 'g')
plt.plot([0], [0], '*')

plt.show()
