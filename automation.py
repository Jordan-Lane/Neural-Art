import random

from numpy_image_nn import NumpyArtGenerator
from file_util import save_numpy_image

if __name__ == "__main__":
    resolution = (5120, 3200)
    color = True

    activation_list = ["tanh", "relu", "sigmoid", "sech"]

    total_images = 10
    for image_num in range(total_images):
        seed = random.randint(1, 4294967294)
        num_layers = random.randint(2, 70)
        hidden_layer_size = random.randint(4, 15)

        print("Image: " + str(image_num) + "/" + str(total_images))

        activation = activation_list[random.randrange(0, len(activation_list))]

        generator = NumpyArtGenerator(resolution, seed, num_layers, hidden_layer_size, activation, color)
        image_result = generator.forward_prop()

        filename = str(generator) + ".jpg"
        save_numpy_image(image_result, filename)