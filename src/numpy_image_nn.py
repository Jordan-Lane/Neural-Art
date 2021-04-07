import argparse
import cv2
import numpy
import numpy_activations as activation_funcs
import os
import random

from file_util import save_numpy_image
from inspect import getmembers, isfunction
#from numpy_activation import *

class NumpyArtGenerator:
    """ Generates imagery using randomness and a fully connected forward propagation network """

    def __init__(self, resolution, seed, num_layers, layer_width, activation_name):
        """Initialize the network.

            resolution          -- tuple resolution of the output network 
            seed                -- seed value used by numpy random 
            num_layers          -- the number of hidden layers in the neural network
            layer_width         -- the number of perceptrons in each hidden layer 
            activation_name     -- name of the activation function used in each hidden layer
        """
        self.resolution = resolution

        self.seed = seed
        numpy.random.seed(self.seed)

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.output_layer = 3

        self.__set_activation(activation_name)


    def __str__(self):
        """String representation of the network. """

        activation_string = self.activation[0]
        return "-".join([str(self.seed), str(activation_string), str(self.num_layers), str(self.layer_width)])


    def __set_activation(self, activation_name):
        """Set the activation functions for the network. """

        activations_dict = {}
        for func_name, func in getmembers(activation_funcs, isfunction):
            activations_dict[func_name] = func

        if activation_name not in activations_dict:
            raise KeyError("Invalid activation function: " + activation_name + ". Supported activation functions can be found in numpy_activation.py.")

        activation_func = activations_dict[activation_name]
        self.activation = (activation_name, activation_func)


    def __generate_input(self):
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
                W = numpy.random.randn(results.shape[1], self.layer_width)

            activation_func = self.activation[1]
            results = activation_func(numpy.matmul(results, W))

        # TODO: Consider actual normalization (pros: more flexible with activation function... cons: probabaly a lot slower)
        results = (1 + results)/2.0
        results = (255.0 * results.reshape(nrows, ncols, results.shape[-1])).astype(numpy.uint8)
        return results


    def run(self):
        inputs = self.__generate_input()
        return self.forward_prop(inputs)


def single_img_generation(args):
    if args.seed is None:
        args.seed = random.randint(seed_min, seed_max)
    if args.layers is None:
        args.layers = random.randint(layers_min, layers_max)
    if args.width is None:
        args.width = random.randint(width_min, width_max)

    if args.resolution is None:
        args.resolution = default_resolution
    else:
        args.resolution = tuple(args.resolution)

    generator = NumpyArtGenerator(args.resolution, args.seed, args.layers, args.width, args.activation)
    numpy_image = generator.run()

    filename = str(generator) + ".jpg"
    output_directory = "../images"
    save_numpy_image(numpy_image, filename, output_directory)


if __name__ == "__main__":
    seed_min = 0
    seed_max = 2147483647
    layers_min = 0
    layers_max = 50
    width_min = 0
    width_max = 20
    default_resolution = (1920, 1080)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    single_parser = subparsers.add_parser('single')
    single_parser.add_argument("-s", "--seed", type=int, help="Seed value used by numpy random. Default is a random value between " + str(seed_min) + " - " + str(seed_max))
    single_parser.add_argument("-l", "--layers", type=int, help="Number of hidden layers. Default is a random int between " + str(layers_min) + " - " + str(layers_max))
    single_parser.add_argument("-w", "--width", type=int, help="Number of perceptrons in each hidden layer. Default is a random int between " + str(width_min) + " - " + str(width_max))
    single_parser.add_argument("-r", '--resolution', nargs=2, type=int, help="Resolution of output image. Default is " + str(default_resolution))
    single_parser.add_argument("-a", "--activation", default="tanh", help="Activation function name used in every hidden layer. Activation functions can be found in the numpy_activation file.")
    single_parser.set_defaults(func=single_img_generation)
    
    args = parser.parse_args()

    # TODO: Check that a func exists and display useful dialog to user
    args.func(args)



