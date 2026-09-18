+++
title = "The Basic Idea Behind Artificial Intelligence"
description = "What even is an Artificial Intelligence?"
date = "2026-09-18"
+++


Here, we will provide an overview of the underlying logic of machine learning.

<!--more-->

## What even is an Artificial Intelligence?

### **Regression**
Imagine a handful of dots scattered across a plane.

<!-- ![[../../Attachements/Informatique/LinearDatapoints.png|418]] -->

{{< figure
  src="/images/LinearDatapoints.png"
  alt="Linear Datapoints"
  link=""
>}}

These data points may be a measurement of a physics experiment, e. g. the price of a house which only depends on its living space, or whatever you want it to be. The point is that you got a variable $y$ which is only dependent on one single variable $x$. Most importantly, they happen to form a sort of *linear relation*; you can clearly see that all your data pools (with some variance) around a linear function with an increase of $1$ and no offset.

Now you are given the task to evaluate your data and make a prediction on a specific $x$ value that you have not measured yet. The most obvious approach is to find a function that approximates the points, so that the mean distance to all points is minimal.
This is what we refer to as a *regression*. In our case it is a linear one.


<!-- ![[../../Attachements/Informatique/LinearRegression.png|431]] -->

{{< figure
  src="/images/LinearRegression.png"
  alt="Linear Regression"
  link=""
>}}

This small example already demonstrates the very basic idea of an *Artificial Intelligence (AI)*. You have some data points which we can already call our *training data*, and we aim to approximate them using a function to make a prediction.

Nevertheless there are two major differences to an actual AI. Firstly, there can be an unlimited number of in- and outputs, not just one of each. You can imagine it as a somewhat more complex function $f$ that takes multiple inputs and gives you multiple outputs. Mathematically speaking, we can write those sets of in- and outputs as vectors. For most people the word "vector" implies a connection to dimensional space, and this may be right for some explanatory analogies, in which spacial representation of data or some process inside the AI actually makes sense, but for this content just consider it a notational trick that helps us to represent and calculate a list of numbers.

$$
f: \mathbb{R}^n \rightarrow \mathbb{R}^m,
\quad
n, m \in \mathbb{N},
\qquad
f(\vec{in}) = \vec{out},
\quad
\text{mapping \(n\) inputs to \(m\) outputs}
$$

Here $f(x)$ is left as a black box for now, but of course we will get to that later on.

Secondly, the inner function is, as mentioned, way more complex, and it is not always clear how it will look in the end. The AI model is given a *model architecture* (basically how the inner function $f$looks) and tries to alter its parameters by looking at different examples of data, so that it can finally (and hopefully) make a useful prediction on a request that is not already included in the training data. This "learning" process is called *training* and is why we are even talking about an "intelligence".


#### **Classification**
The most common usecase for the process described above is a so called *classification* in which you do not want to predict a specific value like the price of a house, but rather try to identify your input as an object out of a given list of classes. For example, a well-known task is the distinction between animals shown on a photograph.

Let us assume that you want to know whether your image shows a dog, a cat or a mouse. These are our three classes and we want to categorise our input and assign it to one of them. Also, for simplicity we assume that only one of the three animal types is present at the same time. In this case you would take the color values of each pixel of the image as the input and plug it into our "magic" function $f$. It is still not important what that function does, because we just briefly want to address how the output would look like.

What we want is a probability value for every possible class (in our case 3), so that the predicted class has the closest value to 1.

$$
\vec{out} \in \mathbb{R}^3_+,
\quad
\vec{out} = [0.15, 0.8, 0.05]
$$

This output would predict the second class, because it has the highest value. Which animal it belongs to would be defined during training; it may be a cat, a dog or a mouse.
For $k$ classes you can write it like that:

$$
\vec{out} \in \mathbb{R}^k_+,
\quad
\vec{out} = [out_0, ..., out_{k-1}],
\qquad
\sum_{i=0}^{k-1} \vec{out_i} = 1
$$

The sum of all elements is $1$ because we are talking about a probability distribution, and we iterate to $k-1$ because we start at $0$. Additionally the index of the highest value is our predicted class.

The obvious question now is how this function $f$ looks, which parameters it consists of, and how we adjust them to give us the desired output.

There are a lot of different model architectures (which, again, just means how $f$ looks mathematically) for a lot of different tasks. There are of course functions that can approximate some data better than others. This is why a *Large Language Model* such as *ChatGPT* looks different from an image classifier. One tries to find patterns in sequential data like language, while the other is trying to understand spacial dependencies in an image, for example which parts of the body have to be present and how are they relatively placed to each other, so that it is a dog and not a cat.

Despite there being a lot more variety and depth on the right model architecture, there is one type that stands out with its simplicity, generality and universal usefulness: the *Neural Network*.

It is one of the earliest types of Artificial Intelligence and is still largely used in combination with other network types. We will use it to explain the mathematical concept behind AI in our article [DeMNISTifying Neural Networks](/post/demnistifying-neural-networks/ "DeMNISTifying Neural Networks")
