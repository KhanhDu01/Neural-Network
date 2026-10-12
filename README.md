# Neural Network
Creating a convolutional neural network (cnn) from scratch using python 
The model was trained on 7000 samples across 345 categories of drawing, taking roughly 6 hours. 
The samples are from https://github.com/googlecreativelab/quickdraw-dataset

The model takes a small 28×28 drawing and processes it through two convolutional layers. These layers learn to detect visual patterns, like edges and simple shapes, while max-pooling reduces the image’s size and keeps the strongest features. The extracted features then pass through dense layers, which produce scores for 345 drawing categories.

During training, the model compares its predictions with the correct labels and adjusts its weights to improve. I implemented the main building blocks in Python with NumPy, including the convolution, pooling, forward and backward passes, and Adam-based weight updates. 

## Try it in the browser

Try on https://khanhdu01.github.io/Neural-Network/

## Note
The UI of the page is made with AI. The main purpose of this repo was to learn and build a neural network from scratch.