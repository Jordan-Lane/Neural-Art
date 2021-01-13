import numpy

def tanh(matrix):
    return numpy.tanh(matrix)

def sigmoid(matrix):
    return 1.0 / (1 + numpy.exp(-matrix))

def relu(matrix):
    return numpy.maximum(matrix, 0, matrix)

def softmax(matrix):
    expo = numpy.exp(matrix)
    expo_sum = numpy.sum(numpy.exp(matrix))
    return expo/expo_sum

def sech(matrix):
    return 1.0 / numpy.cosh(matrix)

# Meta programming idea: Use https://www.tutorialspoint.com/How-to-list-all-functions-in-a-Python-module
#   to create a dict of names -> functions for use in the numpy_image activation layer