+++
title = "DeMNISTifying Neural Networks"
description = "The fundamental mechanism behind Neural Networks"
date = "2026-09-17"
+++

In this article we will try to explain the fundamental mechanism behind Neural Networks by discussing and showing off the classification of the very well known MNIST dataset from the *National Institute of Standards and Technology*. It is a set of handwritten digits in a 28 by 28 grid of pixels, where each pixel has a value between 0 and 255, representing a tone of grey (0 = black, 255 = white). In practice every value is divided by 255, so that we only work with percentual values between 0 and 1.

<!--more-->

We assume that you are coming from this article right [here](/post/the-basic-idea-behind-artificial-intelligence "The Basic Idea Behind Artificial Intelligence"). If you have not read it yet, we highly recommend to catch up on it before continuing.

### **MNIST Dataset**

Example of the number nine:

{{< figure
  src="/images/MNIST_9.png"
  alt="MNIST 9"
  link=""
>}}


The dataset preserves 60.000 labeled images to train on and another 10.000 labeled images to test the models accuracy on unseen data.
Our task will be to train a Neural Network with these digits so that it will be able to identify the digits from $0$ to $9$, with the same format, and tell us which one it is.

### **The Neural Network**
The development of *Neural Networks* goes back to the 1940s. In 1943 *Warren S. McCulloch* and *Walter Pitts* published their paper *"A Logical Calculus of the Ideas Immanent in Nervous Activity"* which introduced the idea of a mathematical concept imitating the functionality of biological nerve systems, like the human brain. Furthermore, the concept should be used to solve problems that cannot be described directly in simple rules, but can be worked out by analysing a given set of data and extracting its patterns autonomously.

In more direct terms this means that you do not define your mathematical function from the beginning but rather define its outer form and change its inner parameters algorithmically by looking at different data examples.

This technique was developed ever since and laid the foundation for the modern *AI* used today.
Of course nowadays, there are a lot of different functions that adapted these early developed principles, but a finalized version of the *Neural Network* exists to this day and will be discussed in the following.

## **Training Cycle**
*Neural Networks* are trained in three steps:

#### 1. The Forward Pass

Firstly, the network looks at one or multiple examples of the data and calculates an output. It may be correct or not. One set of inputs (e. g. the pixels of an image) is called a *sample*; multiple samples are referred to as a *batch*. It is indeed possible to process multiple samples, a batch, at once to decrease computing time.

At the end of the *forward pass* the network calculates the *loss* or *cost* which measures how far the networks prediction is from the wanted output. The closer the loss is to $0$, the better the prediction.

#### 2. The Backward Pass

Secondly, the network calculates, from the back to the front, the influence of every inner parameter on the prediction using the calculated loss. It figures out which parameters were responsible for it to fail or being less precise.

#### 3. Stochastic Gradient Descent
Finally, using a method called *Stochastic Gradient Descent* <abbr title="Stochastic Gradient Descent">SGD</abbr>, every parameter is changed based on its calculated influence. The goal is that the same batch would now result in a better prediction. Therefore, if all data samples of the data set follow the same patterns, the network will slowly learn to make better predictions on the given type of data.

The math behind this training concept will furthermore be discussed.

# **The Forward Pass**

### **Connecting Neurons Mathematically**
This is one of the most common representations of a *Neural Network*:

<!-- ![[../../Pasted image 20260915101906.png|386]] -->

At first glance, this may look scary, but its idea truly is simple.
Every dot represents a *neuron*; the connections between the neurons are referred to as *weights*. Also you may have already noticed the *neurons* being grouped in *layers*. The output of the *Neural Network* is normally calculated step by step by calculating layer after layer from left to right. Every *neuron* of one *layer* is connected to every *neuron* of the previous *layer*, and therefor receives data from all of the previous *neurons*. This is why this structure is sometimes called a *fully interconnected layer*.

To sum all this up mathematically, a *neuron* ${}^Ln_{i}$ of the *layer* $L$ is a weighted sum of all *neurons* ${}^{L-1}n_{i}$ of the previous *layer* $L-1$. Most importantly, each *neuron* ${}^Ln_{i}$ has its own *weight* $w_{i,j}$ for every *neuron* ${}^{L-1}n_{j}$, so that its interprets them differently than all the other *neurons* of the same *layer*.

The *input layer* $L=0$ is somehow special because it does not calculate anything. Its *neurons* are just set to the input, so that each *neuron* of the layer corresponds to one *feature* of the input *sample*, for example a pixel of an image. Now we notate every layer as a vector that contains its neurons:

$$
^{L}\vec{n} =
\begin{bmatrix}
^{L}n_0 & ^{L}n_1 & \dots & ^{L}n_{m-1}
\end{bmatrix}
\qquad
m \in \mathbb{N},
\quad \text{m is the number of neurons in } ^L\vec{n}
$$

The input of one of the digits of the MNIST dataset, where each of the $28 \times 28$ pixels $p$ is represented by a tone of gray, would therefore look like this:

$$
\begin{aligned}
\vec{in} &=
\begin{bmatrix}
p_0 & p_1 & \dots & p_{28 \cdot 28 - 1}
\end{bmatrix} \\[0.3em]
&= \begin{bmatrix}
^0n_0 & {}^0n_1 & \dots & ^0n_{783}
\end{bmatrix} \\[0.3em]
&=  {}^0\vec{n}
\end{aligned}
$$

Now that we got an input we can begin to calculate our first *layer* $L=1$.
For every *neuron* ${}^1n_i$ the following formula applies:

$$
{}^1n_i = \sum_{j=0}^{783} {}^0\vec{n}_j\vec{w}_j
$$

Here $\vec{w}$ holds all the connections (the weights) between the *neuron* ${}^1n_m$ and all the *neurons* of the first *layer* ${}^0\vec{n}$. Of course $\vec{w}_0$ corresponds to ${}^1\vec{n}_0$, $\vec{w}_1$ to ${}^1\vec{n}_1$ and so on.
The interesting part about this is that you can just rewrite that expression for every *neuron* as a dot product of the *neurons* of the previous *layer* and the *neuron's* *weights*:

$$
{}^1n_i = {}^0\vec{n} \cdot \vec{w}
\qquad
\dim({}^0\vec{n}) = \dim(\vec{w})
$$

We can take this even further by combining every weight vector of every neuron in a layer as a *matrix* like this:

$$
{}^LW =
\begin{pmatrix}
{}^Lw_{0,0} & {}^Lw_{0,1} & \cdots & {}^Lw_{0,m-1}\\
{}^Lw_{1,0} & {}^Lw_{1,1} & \cdots & {}^Lw_{1,m-1}\\
\vdots & \vdots & \ddots & \vdots\\
{}^Lw_{n-1,0} & {}^Lw_{n-1,1} & \cdots & {}^Lw_{n-1,m-1}\\
\end{pmatrix}
\qquad
\begin{aligned}
{}^LW &\in \mathbb{R}^{in, out}\\
n &= \dim({}^{L-1}\vec{n}) \\
m &= \dim({}^{L}\vec{n}) \\
\end{aligned}
$$

Every column now represents all the weights of one neuron of the layer that is being calculated while the every row stands for one neuron of the input of the previous layer.
If we continue this idea and take the dot product of our input vector ${}^{L-1}\vec{n}$ and the weight matrix ${}^LW$ of the current layer, we get the correct value of each neuron according to the graph.

$$
{}^{L-1}\vec{n} \cdot {}^LW
\quad=\quad
{}^{L}\vec{n}
$$

If we have $n$ *neurons* in layer $L-1$ and $m$ *neurons* in *layer* $L$, the dimensions of the containers look like this:

$$
{}^{L-1}\vec{n} \in \mathbb{R}^n
\qquad
{}^{L}W \in \mathbb{R}^{n,m}
\qquad
{}^{L}\vec{n} \in \mathbb{R}^m
$$

This means that we are mapping $n$ input dimension to $m$ output dimensions.
The explained process is then continued with every layer of the *Neural Network*. The amount of layers and neurons in each layer can be defined freely; there is no right or wrong, but some structures of course work better for some data than others do. The only exceptions are the input layer which needs the exact amount of neurons of your measured features, and the output layer which also needs to have the right amount of neurons. The reason for that will be covered in another chapter.

In practice, this means that every *layer* owns a weight matrix that indicates all connections to the previous layer. The data example is then encoded as a vector which is furthermore *passed forward* through the network. Every layer, one after another, multiplies the output vector of the previous layer with its internal weight matrix and passes on its own output vector to the next layer.

By the way, the *weights* are one of the two types of parameters inside the network that change during the training process. They are adjusted so that the network can make valid predictions.

### **Adding Bias**
In addition to the weights every neuron contains a *bias*, that basically is a constant which is added to the weighted sum of the input neurons. This constant ensures that the network is able to shift all inputs and can cover a wider numerical range. Again, we can bundle all biases of one layer $L$ in a vector $\vec{b}$.

$$
{}^L\vec{b} \in \mathbb{R}^{n}
\qquad
n \in \mathbb{N},
\quad
\text{which is the number of neurons in the layer L}
$$

Now ${}^L\vec{b}$ is just added to the weighted sum to calculate the layer output:

$$
{}^{L-1}\vec{n} \, \cdot \, {}^LW + {}^L\vec{b} \, = {}^{L}\vec{n}
$$

The bias is trained during training too.

### **The Output**
As mentioned in the introduction, we need a vector with an element for every class that can possibly be predicted by the network. All elements together form a probability distribution and the index of the highest value indicates the predicted *class*.

The part about the probability distribution is not important yet, but we already gained the knowledge to understand how the dimensionality of the output comes into place. And it is fairly simple:
We just define the last *layer* of the *Neural Network* to have one *neuron* for every *class*. Mathematically speaking, this means that the last layer $L$ multiplies the output vector of the *layer* $L-1$ with a matrix of the dimensions $\text{inputs x classes}$ and hence mapping the input to the dimension we need.

### **But why does this work?**
Up to this point we only discussed the mathematical notation of a *Neural Network*. But why should the concept shown on the graph even be able to find patterns in data and, for example, understand what is shown on an image?

Thinking of our current example data of handwritten digits, an ideally set up network could work like this:
Every neuron in the second layer, so the first layer after the input layer, looks at different neurons of the input. A crucial characteristic of the weights between the neurons is, that they can become close to zero during training. As a consequence neurons can learn to ignore specific neurons of their input or weight other neurons more if we think the other way around.

Thus it would be logical for every neuron of a layer to look at a specific area of the image and tell for example whether there is a line or a dot in the top right or the top left corner. If so, the neuron responsible for that would output a high value because in our data white is defined as a high value close to $1$.

The following layer then would continue to group these clusters even more. Does one line extend on another? Do they form a circle? Of course the exact process of that depends on the amount of layers and the number of neurons in each of them, but in general this is how you can interpret the passing of information through the layers.

Every layer of neurons adds a new layer of abstraction of the image. All pixels are clustered differently again and again until, in the end, the network can make a decision by looking at all the present features in the image.

### **The Activation Function**
The described mechanism can also be interpreted as a *regression*, which it basically is. We


{{< figure
  src="/images/Activation_ReLU.png"
  alt="ReLU"
  link=""
>}}


### **The Last Layer**

### **The Loss Function**

# **The Backward Pass**
