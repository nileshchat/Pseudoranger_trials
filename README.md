# Pseudoranger_trials
The code is incomplete for now, however the model is running fine. The model has two intermediate layers.
Keras uses theano for its backend.

## Linear Regression
The X's and Y's are estimated separately. And then plotted against the jittery orbit. The results however uptill now are not satisfactory.

## Convolutional Filter
The Convolutional averaging filter performs much better as compared to the previous regression based model.
![Convolutional Filter](/Models/conv.png)

## Exponential Smoothing
The model performs exceptionally for some values of smoothing constant.

|![Smoothing Rate = 1](http://pixady.com/image/niz/) | ![Smoothing Rate = 0.5](http://pixady.com/image/nj0/)|
|----------------------------------------------------|------------------------------------------------------|
|![Smoothing Rate = 0.3](http://pixady.com/image/nj1/)|![Smoothing Rate = 0.1](http://pixady.com/image/nj2/)|

The code is still in development phase. More models are being tried.

## Credits
I would like to thank @hornig for [orbit generation code](https://github.com/Nilesh4145/Pseudoranger_trials/blob/master/generate_orbit.py) which can be found in this repository -> [a03-LonePseudorangerOrbitPosition](https://github.com/aerospaceresearch/summerofcode2017/tree/master/gsoc2017/a03-LonePseudorangerOrbitPosition)
