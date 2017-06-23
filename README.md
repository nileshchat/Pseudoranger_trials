# Pseudoranger_trials
This code was made to test various noise cancellation filters for smoothing orbital data.

## Linear Regression
The X's and Y's are estimated separately. And then plotted against the jittery orbit. The results however uptill now are not satisfactory.

## Convolutional Filter
The Convolutional averaging filter performs much better as compared to the previous regression based model.
![Convolutional Filter](/Models/conv.png)

## Exponential Smoothing
The model performs exceptionally for some values of smoothing constant.

|![Smoothing Rate = 1](http://img.pixady.com/2017/03/168166_1_460x312.png) | ![Smoothing Rate = 0.5](http://img.pixady.com/2017/03/166047_2_460x326.png)|
|----------------------------------------------------|------------------------------------------------------|
|![Smoothing Rate = 0.3](http://img.pixady.com/2017/03/337461_3_460x334.png)|![Smoothing Rate = 0.1](http://img.pixady.com/2017/03/467563_4_460x336.png)|

## Credits
I would like to thank @hornig for [orbit generation code](https://github.com/Nilesh4145/Pseudoranger_trials/blob/master/generate_orbit.py) which can be found in this repository -> [a03-LonePseudorangerOrbitPosition](https://github.com/aerospaceresearch/summerofcode2017/tree/master/gsoc2017/a03-LonePseudorangerOrbitPosition)
