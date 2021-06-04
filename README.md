# Visualiation-of-Loss-Function
Tensorflow implementation of one of the methods found in this paper https://arxiv.org/abs/1712.09913

The paper, Visualizing the Loss Landscape of Neural Nets(2017) by Li , provides a variety of methods for visualizing high
dimensional loss functions.The paper provides a github repository containing the described methods. The methods in that repository(https://github.com/tomgoldstein/loss-landscape) are written in Pytorch. Since I 
use tensorflow, I wrote the method that I needed in Tensorflow and provided it here in this repository.

This github respository is a Tensorflow based implementation of the "filter-normalization" method introduced in section 4 of the paper. 

But to reiterate from the paper, the idea is to train your model until the loss function converges and then choose two random vectors from a multivariate normal distribution of the same dimension as your weights. You vary your model in the direction of these two vectors taking scalar multiples of the vector between two scalar bounds. Then, you plot the outcome of the loss function over a test or training dataset. The idea is to see what the convexity of 2 random directions of your loss function looks like. Is it highly non-convex with many singularities or is it smooth? As the paper suggests, this does not give you the complete picture, but is a test to gain new information about how your loss function is behaving near a local minima.


Dependencies:
Tensorflow 2.4 ,
numpy 1.19.5 ,
matplotlib 3.3.4 ,

               A few examples of the output can be seen below.
![image](https://user-images.githubusercontent.com/35053174/120835087-7f007700-c529-11eb-949c-8c8c6a9f6ffd.png)
