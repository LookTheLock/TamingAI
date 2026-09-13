# CGI

The input, here a bitmap of an image of a handwritten digit, comes from the clinet side.
The shell script gets it as `stdin`,
then pipes it to the `NeuralNetwork` executable which outputs the result in `stdout`.
The output is one number, that being the digit of the input image.
