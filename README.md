This is a part from the book """Deep learning for Computer vision, Starter Bundle"""

The VGG family of Convolutional Neural Networks can be characterized by two key components:
1. All CONV layers in the network using only 33 filters.
2. Stacking multiple CONV => RELU layer sets (where the number of consecutive CONV =>
RELU layers normally increases the deeper we go) before applying a POOL operation


MiniVGGNet consists of two sets of CONV => RELU => CONV => RELU => POOL
layers, followed by a set of FC => RELU => FC => SOFTMAX layers. The first two CONV layers
will learn 32 filters, each of size 33. The second two CONV layers will learn 64 filters, again, each
of size 33. Our POOL layers will perform max pooling over a 22 window with a 22 stride.
We’ll also be inserting batch normalization layers after the activations along with dropout layers
(DO) after the POOL and FC layers.

we use MaxPooling2D with a size of 22. Since we do not explicitly set a stride,
Keras implicitly assumes our stride to be equal to the max pooling size (which is 22).
We then apply Dropout with a probability of p = 0:25, which this implies that a
node from the POOL layer will randomly disconnect from the next layer with a probability of 25%
during training. We apply dropout to help reduce the effects of overfitting.


Our FC layer has 512 nodes, which will be followed by a ReLU activation and BN. We’ll
also apply dropout here, increasing the probability to 50% – typically you’ll see dropout with
p = 0:5 applied in between FC layers.
