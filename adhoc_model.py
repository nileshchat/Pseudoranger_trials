import csv
import numpy
from keras.layers import Dense, Input
from keras.models import Model, Sequential
import matplotlib.pyplot as plt

reader = csv.reader(open("track1489492822.csv", "r"), delimiter = ",")

l = list(reader)
x_train = []
x_res = []

y_train = []
y_res = []


for i in l[1:]:
	x_train.append(i[4])
	x_res.append(i[1])
	y_train.append(i[5])
	y_res.append(i[2])

X_train = numpy.array(x_train).astype("float64")
X_res = numpy.array(x_res).astype("float64")
Y_train = numpy.array(y_train).astype("float64")
Y_res = numpy.array(y_res).astype("float64")


model1= Sequential()
model1.add(Dense(12, input_dim=1, init='uniform', activation='linear'))
model1.add(Dense(1, init='uniform', activation='linear'))

model1.compile(loss='cosine_proximity', optimizer='sgd')
model1.fit(X_train, X_res, nb_epoch=50, batch_size=1)

new_reader = csv.reader(open("track1489609280.csv", "r"), delimiter = "\t")
l1 = list(new_reader)

test_x = []
test_y = []

for i in l1[1:]:
	test_x.append(i[4])
	test_y.append(i[5])

Test_x = numpy.array(test_x).astype("float64")
Test_y = numpy.array(test_y).astype("float64")

pred_x = model1.predict(Test_x)

#model1.fit(Y_train, Y_res, nb_epoch=50, batch_size=1)
#pred_y = model2.predict(Test_y)

a_x = []
a_y = []
for i in range(len(pred_x)):

	a_x.append(pred_x[i]+Test_x[i])
	a_y.append(pred_x[i]+Test_y[i])

fig = plt.figure()
sub1 = fig.add_subplot(221)
sub1.set_title('Un-filtered Orbit plot')
sub1.plot(test_x)

sub2 = fig.add_subplot(223)
sub2.set_title('Filtered Orbit plot')
sub2.plot(a_x)

plt.tight_layout()
plt.show()