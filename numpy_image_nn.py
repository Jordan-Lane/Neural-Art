import cv2
import numpy
import numpy_activation as activation
import os

from file_util import save_numpy_image
from inspect import getmembers, isfunction
#from numpy_activation import *

class NumpyArtGenerator:
    def __get_activation(self, activation_string):
        activations_dict = {}

        for func_name, func in getmembers(activation, isfunction):
            activations_dict[func_name] = func

        if activation_string not in activations_dict:
            raise KeyError("Activation: " + activation_string + " not found in activation functions")
        
        activation_func = activations_dict[activation_string]
        return (activation_string, activation_func)


    def __init__(self, resolution, seed, num_layers, hidden_layer_size, activation_string, color):
        self.resolution = resolution
        self.color = color

        self.seed = seed

        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = self.__get_activation(activation_string)
        
        if color:
            self.output_layer = 3

        if seed:
            numpy.random.seed(seed)

    def __str__(self):
        color_string = "RGB" if self.color else "BW"
        activation_string = self.activation[0]
        return "-".join([str(self.seed), str(activation_string), str(self.num_layers), str(self.hidden_layer_size), color_string])

    def generate_input(self):
        (ncols, nrows) = self.resolution

        rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(min(nrows, ncols)/2.0)
        colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(min(nrows, ncols)/2.0)

        inputs = [rowmat, colmat, numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
        return numpy.stack(inputs).transpose(1, 2, 0).reshape(-1, len(inputs))


    def forward_prop(self, inputs):
        results = inputs.copy()

        (ncols, nrows) = self.resolution

        for layer in range(0, self.num_layers):
            print("Generating Image. Current neural layer: " + str(layer+1) + "/" + str(self.num_layers), end='\r')

            if layer == self.num_layers - 1:
                W = numpy.random.randn(results.shape[1], self.output_layer)
            else:
                W = numpy.random.randn(results.shape[1], self.hidden_layer_size)

            activation_function = self.activation[1]
            results = activation_function(numpy.matmul(results, W))

        results = (1 + results)/2.0
        results = (255.0*results.reshape(nrows, ncols, results.shape[-1])).astype(numpy.uint8)

        return results


if __name__ == "__main__":
    resolution = (5120, 3200)
    seed = 8769999
    num_layers = 8
    hidden_layer_size = 8
    color = True
    activation_string = "tanh"

    generator = NumpyArtGenerator(resolution, seed, num_layers, hidden_layer_size, activation_string, color)

    input_values = generator.generate_input()
    result = generator.forward_prop(input_values)

    filename = str(generator) + ".jpg"
    save_numpy_image(result, filename)

    