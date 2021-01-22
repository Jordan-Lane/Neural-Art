import random

from numpy_image_nn import NumpyArtGenerator
from file_util import save_numpy_image

class ImageGenAutomation():
    """Automation script for generating multiple images from the NumpyArtGenerator class. """

    def __init__(self, resolution, color, number_of_images):
        self.number_of_images = number_of_images
        self.color = color
        self.resolution = resolution

        self.possible_activations = [("tanh", 50), ("relu", 20), ("sigmoid", 1), ("sech", 1)]

        self.seed_range = (1, 4294967294)
        self.num_layers_range = (2, 120)
        self.hidden_layer_size_range = (3, 24)


    def __generate_activations(self, multiple_activation_functions, num_layers):
        activation_strings, activation_weights = list(zip(*self.possible_activations))

        activations = []
        if multiple_activation_functions:
            for _ in range(num_layers):
                activation_choice = random.choices(activation_strings, weights = activation_weights)[0]
                activations.append(activation_choice)
        else:
            activations.append(random.choice(activation_strings))

        return activations

    
    def run(self):
        for image_num in range(self.number_of_images):
            seed = random.randint(*self.seed_range)
            num_layers = random.randint(*self.num_layers_range)
            hidden_layer_size = random.randint(*self.hidden_layer_size_range)

            multiple_activation_functions = random.choice([True, False])
            activations = self.__generate_activations(multiple_activation_functions, num_layers)

            activation_display_string = "mixed" if multiple_activation_functions else activations[0]

            print("Image: " + str(image_num) + "/" + str(self.number_of_images))
            print("   |-> Seed: " + str(seed))
            print("   |-> Hidden Layer Size: " + str(hidden_layer_size))
            print("   |-> Num Layers: " + str(num_layers))
            print("   |-> Activation: " + activation_display_string)

            art_gen = NumpyArtGenerator(self.resolution, self.color, seed, num_layers, hidden_layer_size, activations)
            image_result = art_gen.run()

            filename = str(art_gen) + ".jpg"
            save_numpy_image(image_result, filename)


def main():
    resolution = (1656, 3584)
    color = True
    number_of_images = 50

    generator = ImageGenAutomation(resolution, color, number_of_images)
    generator.run()


if __name__ == "__main__":
    main()