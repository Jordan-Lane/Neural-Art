import argparse
import random
from art_net import NumpyArtGenerator
from file_util import save_numpy_image

seed_min = 0
seed_max = 2147483647
layers_min = 0
layers_max = 50
width_min = 0
width_max = 20
default_resolution = (1920, 1080)


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
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    single_parser = subparsers.add_parser('single')
    single_parser.add_argument("-s", "--seed", type=int, help="Seed value used by numpy random. Default is a random int between " + str(seed_min) + " - " + str(seed_max))
    single_parser.add_argument("-l", "--layers", type=int, help="Number of hidden layers. Default is a random int between " + str(layers_min) + " - " + str(layers_max))
    single_parser.add_argument("-w", "--width", type=int, help="Number of perceptrons in each hidden layer. Default is a random int between " + str(width_min) + " - " + str(width_max))
    single_parser.add_argument("-r", '--resolution', nargs=2, type=int, help="Resolution of output image. Default is " + str(default_resolution))
    single_parser.add_argument("-a", "--activation", default="tanh", help="Activation function used in every hidden layer. Activation functions can be found in the numpy_activation file.")
    single_parser.set_defaults(func=single_img_generation)
    
    args = parser.parse_args()

    # TODO: Check that a func exists and display useful dialog to user
    args.func(args)

