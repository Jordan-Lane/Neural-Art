&nbsp;
# Neural Art

Neural Art is a command line tool designed for fast and easy generation of Compositional Pattern Producing Networks. The toll can be used to create either a single, or batch of random images using the underlying network. 

The Numpy CPPN used to generate the image is directly based on the [blog post by Nenad Markuš](https://nenadmarkus.com/p/visualizing-audio-with-cppns/). I highly recommend reading the blog post as it gives a great introduction to CPNNs and the underlying math achieved with numpy. The post also goes on to create a very cool audio visualizer using CPPNs and Fourier Transforms. 

Sample high resolution examples can be found in the /images/ folder. 

&nbsp;
## Installation

This project is built using Python 3.9 and Pipenv. If you are unfamiliar with Pipenv, I recommend reading the official [Pipenv documentation](https://pipenv-fork.readthedocs.io/en/latest/).

After cloning the project, navigate into the root directory and run:

```zsh
pipenv shell
pipenv sync
```

This will create a new Python 3.9 virtual environment and install all neccesary dependencies from Pipfile.lock. If you have any issues while installing please refer to the **Common Installation Issues** at the bottom of this README.

&nbsp;
## Usage
The tool can be used to generate either a single, or a batch of images. The single mode is designed to have a finer control over the underlying CPPN structure, while the batch mode is for generating multiple completely random images. 

### Single Image Creation
```zsh
python main.py single [-h] [-v] [-s SEED] [-l LAYERS] [-w WIDTH] [-r RESOLUTION RESOLUTION] [-a ACTIVATION]
```

The single command generates a single abstract image based on the neural network settings passed in by the various flags. All flags are optional, and have default values. 

More info about the various flags and their default values can be found with the help flag:
```zsh
python main.py single -h
```

#### Example
Generate a completely random image with a default resolution of 1920x1080.
```zsh
python main.py single
```

Generate an image with a resolution of 5120x3600, numpy seed of 444, and a CPPN with 10 layers, width of 8 perceptrons, and a tanh activation function. 
```zsh
python main.py single -v -r 5120 3600 -s 444 -l 10 -w 8 -a tanh
```

&nbsp;
### Batch Image Creation
```zsh
python main.py batch [-h] [-v] [-r RESOLUTION RESOLUTION] [-a ACTIVATION] number_of_images
```

The batch command generates multiple images with random CPPN settings. The only mandatory arguement is `number_of_images`, which specifies the number of images to be generated. 

#### Example
Generate 10 random images using the sech activation function
```zsh
python main.py batch -a sech 10
```

&nbsp;
## Tips and Tricks
This tool is designed for experimentation! Therefore I encourage you to mess around with the different flags and the effects they have on the generated image.

A couple things:
- I recommend setting the resolution quite high. The image generation will be quite a lot slower, but the outputted images are generally much nicer (YMMV).
- Tanh is by far the best activation function for producing interesting art. I plan on adding functionality to use multiple activation functions, however I haven't figured out a nice way to achieve this with the current CLI. 
- Have fun!

&nbsp;
## Common Installation Issues
If unable to build numpy wheel for opencv-python package, you need to set the environment variable SYSTEM_VERSION_COMPAT=1. https://github.com/pypa/pipenv/issues/4576. 

&nbsp;
## References
Original Numpy network used here:
[Using CPPNs to generate abstract visualizations from audio data](https://nenadmarkus.com/p/visualizing-audio-with-cppns/)

Further CPPN Reading and Projects:
[Compositional Pattern Producing Networks: A Novel Abstraction of Development](https://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf)
[Understanding Compositional Pattern Producing Networks (Part One)](https://towardsdatascience.com/understanding-compositional-pattern-producing-networks-810f6bef1b88)
[Real time abstract art generation using a neural net](https://www.expunctis.com/2020/01/19/Abstract-art.html)
[Generating Abstract Patterns with Tensorflow](https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/)

