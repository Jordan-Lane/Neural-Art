import cv2
import numpy
import numpy_activation as activation
import os

from file_util import save_numpy_image
from inspect import getmembers, isfunction
#from numpy_activation import *

class NumpyArtGenerator:
    """ Generates imagery using randomness and a fully connected forward propagation network """


    def __init__(self, resolution, color, seed, num_layers, hidden_layer_size, activation_list):
        """Initialize the network.

            resolution          -- tuple resolution of the output network 
            color               -- Boolean describing if output is RGB or Greyscale
            seed                -- seed value used by numpy random 
            num_layers          -- the number of hidden layers in the neural network
            hidden_layer_size   -- the number of perceptrons in each hidden layer 
            activation_list     -- a list of activation function names used in the network. The list can have a length of:
                                    -> len(activation_list) == 1              the same activation function is used on all layers
                                    -> len(activation_list) == num_layers     each layer receives its own activation functions
        """
        self.resolution = resolution
        self.color = color

        self.seed = seed

        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        
        self.__set_activation(activation_list)

        if color:
            self.output_layer = 3
        else:
            self.output_layer = 1

        if seed:
            numpy.random.seed(seed)


    def __str__(self):
        """String representation of the network. """

        color_string = "RGB" if self.color else "BW"

        if len(self.activations) == 1:
            activation_string = self.activations[0].__name__
        else:
            activation_string = "mixed"

        return "-".join([str(self.seed), str(activation_string), str(self.num_layers), str(self.hidden_layer_size), color_string])


    def __set_activation(self, activation_list):
        """Set the activation functions for the network. """

        if not isinstance(activation_list, list):
            raise TypeError("Activation_list is incorrect type. Expected: <class 'list'> Received: " + str(type(activation_list)))
        elif len(activation_list) != 1 and len(activation_list) != self.num_layers :
            raise ValueError("Activation_list is incorrect length. Length of the list must either be 1, or the number of hidden layers in the network.")

        activations_dict = {}
        for func_name, func in getmembers(activation, isfunction):
            activations_dict[func_name] = func

        self.activations = []
        for activation_string in activation_list:
            if activation_string not in activations_dict:
                raise KeyError("Activation: " + activation_string + " not found in activation functions")
        
            activation_func = activations_dict[activation_string]
            self.activations.append(activation_func)


    def __get_activation_func(self, layer_num):
        """Get the activation function for the given hidden layer. """

        if len(self.activations) == 1:
            index = 0
        else:
            index = layer_num

        return self.activations[index]


    def __generate_coordinate_input(self):
        """Generate the x,y coordinate matrices used as input for the network. """
        (ncols, nrows) = self.resolution

        rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(min(nrows, ncols)/2.0)
        colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(min(nrows, ncols)/2.0)

        inputs = [rowmat, colmat, numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
        return numpy.stack(inputs).transpose(1, 2, 0).reshape(-1, len(inputs))


    def forward_prop(self, inputs):
        """Run forward propagation on the network"""
        results = inputs

        (ncols, nrows) = self.resolution

        for layer in range(0, self.num_layers):
            print("Current neural layer: " + str(layer+1) + "/" + str(self.num_layers), end='\r')

            if layer == self.num_layers - 1:
                W = numpy.random.randn(results.shape[1], self.output_layer)
            else:
                W = numpy.random.randn(results.shape[1], self.hidden_layer_size)

            activation_function = self.__get_activation_func(layer)
            results = activation_function(numpy.matmul(results, W))

        # TODO: Consider actual normalization (pros: more flexible with activation function... cons: probabaly slower)
        # Would be curious to comment out this line and see what happens...
        results = (1 + results)/2.0
        results = (255.0*results.reshape(nrows, ncols, results.shape[-1])).astype(numpy.uint8)
        return results

    def run(self):
        inputs = self.__generate_coordinate_input()
        return self.forward_prop(inputs)


if __name__ == "__main__":
    resolution = (5120, 3200)
    seed = 77704599
    num_layers = 35
    hidden_layer_size = 5
    color = True

    activation_list = ["tanh"]
    # Example of multiple activation-functions (len(activation_list) == num_layers)
    # activation_list = ["tanh", "tanh", "tanh", "tanh", "tanh", "tanh", "sech", "tanh", "tanh"]

    generator = NumpyArtGenerator(resolution, color, seed, num_layers, hidden_layer_size, activation_list)
    image_result = generator.run()

    filename = str(generator) + ".jpg"
    save_numpy_image(image_result, filename)
