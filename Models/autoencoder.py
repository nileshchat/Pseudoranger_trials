# Nilesh Chaturvedi : 15th Mar'17

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_data(filename):

	reader = csv.reader(open(filename, "r"), delimiter = "\t")
	data_list= list(reader)
	x_true = []
	x_noisy = []
	y_true = []
	y_noisy = []

	for i in data_list[1:]:
		x_true.append(i[1])
		x_noisy.append(i[4])
		y_true.append(i[2])
		y_noisy.append(i[5])
	
	return [x_true, x_noisy, y_true, y_noisy]
'''
Define Model
'''
encoding_dim = 10
input_img = Input(shape = (1,))
encoded = Dense(encoding_dim, activation = 'relu')(input_img)
decoded = Dense(1, activation = 'sigmoid')(encoded)
autoencoder = Model(input = input_img, output = decoded)
encoder = Model(input = input_img, output = encoded)
encoded_input = Input(shape = (encoding_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(input = encoded_input, output = decoded_layer(encoded_input))
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

data = load_data("track1489609280.csv")
x = np.array(data[0], dtype = np.float)
xn= np.array(data[1], dtype = np.float)
y = np.array(data[2], dtype = np.float)
yn= np.array(data[2], dtype = np.float)
trueX = [[i] for i in x]
noisyX = [[i] for i in xn]
trueY = [[i] for i in y]
noisyY = [[i] for i in yn]

# x_train = x_train.astype('float32')/255 #normalization
# x_test = x_test.astype('float32')/255
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# print (x_train.shape)
# print (x_test.shape)
# validation_data=(x_test, x_test)
autoencoder.fit(noisyX, trueX,
                nb_epoch=50,
                batch_size=36,
                shuffle=True,
                )
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)

# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()