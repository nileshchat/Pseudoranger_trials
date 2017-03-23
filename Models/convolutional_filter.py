# Nilesh Chaturvedi
# 21 March, 2017

import matplotlib.pylab as plt
import numpy as np
import csv


def load_data(filename):

	reader = csv.reader(open(filename, "r"), delimiter = "\t")
	data_list= list(reader)
	x_comp = []
	y_comp = []

	for i in data_list[1:]:
		x_comp.append(i[4])
		y_comp.append(i[5])
	
	return [x_comp, y_comp]

def filter(data):
	# Component wise convolution
	step=np.ones([7])/7;
	filtered=np.convolve(np.array(data, dtype = np.float), step, mode="full");
	for i in range(0, 5):
		filtered[i] = data[i]
		filtered[len(data)-i] = data[len(data)-1]
	
	return filtered

def write_to_file(filename, list1, list2):
	file  = open(filename+".csv", "w")
	file.write("X\tY\n")
	for i in range(len(list1)):
		entry = str(list1[i])+"\t"+str(list2[i])+"\n"
		file.write(entry)
	file.close()

if __name__ == '__main__':

	coords = load_data("track1489609280.csv")
	filtered_X = filter(coords[0])
	filtered_Y = filter(coords[1])

	write_to_file("filtered_data", filtered_X, filtered_Y)
	
	fig = plt.figure()
	sub1 = fig.add_subplot(221)
	sub1.set_title('Overlapped Figure')
	sub1.plot(coords[0], coords[1], 'b')
	sub1.plot(filtered_X, filtered_Y, 'g')
	sub1.legend(['Noisy Orbit', 'Filtered Orbit'])
	sub2 = fig.add_subplot(223)
	sub2.set_title('Filtered Orbit')
	sub2.plot(filtered_X, filtered_Y, 'g')
	sub2.legend(['Filtered Orbit'])

	plt.tight_layout()
	plt.show()