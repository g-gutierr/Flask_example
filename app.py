# Importing Required Libraries

import os
from os import listdir

import numpy as np

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as ims

from flask import Flask, send_file


from PIL import Image
from IPython import display
from tqdm import tqdm
import matplotlib.image as ims

from keras import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2DTranspose, Reshape, Conv2D, BatchNormalization, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


import tensorflow as tf
from tensorflow import keras

model_path_gen = 'model/model_gen.h5'

noise_dim = 100
image_dim = (32, 32)


def plot_gen(n_ex=1,dim=(4,4), figsize=(7,7) ):
    noise = np.random.normal(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,:]

        # mover esto a una funcion
        img = (img + 1) / 2

        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return img

def fig_response(fig):
    """Turn a matplotlib Figure into Flask response"""
    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    ims.imsave('name.png', img_bytes)
    return send_file(img_bytes, mimetype='image/png')




generator = load_model(model_path_gen)



app = Flask(__name__)


@app.route('/')
def frontpage():
    return """
<!doctype html>
<head><title>dynamic</title></head>
<body>

<div>
<img style="border: 1px dotted red" src="/example1.png" />
</div>

</body>
</html>
"""

@app.route('/example1.png')
def example1():
    test = plot_gen(n_ex=1,dim=(1,1), figsize=(3,3))
    ims.imsave('example1.png', test)
    #fig, ax = plt.subplots()
    #draw1(ax)
    return nocache(fig_response(a))


def draw1(ax):
    """Draw a random scatterplot"""
    x = [random.random() for i in range(100)]
    y = [random.random() for i in range(100)]
    ax.scatter(x, y)
    ax.set_title("Random scatterplot")


#@app.route('/name.png')
#def example1():
#    generator = define_generator(100)
#    generator.summary()

#    generator.load_weights(model_path_gen)
#    test = plot_gen(n_ex=1,dim=(2,2), figsize=(7,7) )
#    test = plot_gen(n_ex=1,dim=(1,1), figsize=(3,3))

#    ims.imsave('name.png', test)
#    fig, ax = plt.subplots()

#    return nocache(fig_response(name))



#@app.route('/example2.png')
#def example2():
#    """Draw a hexbin with marginals
#
#    From https://seaborn.pydata.org/examples/hexbin_marginals.html
#    """
#    sns.set(style='ticks')
#    rs = np.random.RandomState(11)
#    x = rs.gamma(2, size=1000)
#    y = -.5 * x + rs.normal(size=1000)
#    plot = sns.jointplot(x, y, kind='hex', color='#4CB391')
#    fig = plot.fig
#    return nocache(fig_response(fig))


#def plot_image(image):
#  plt.imshow(image)


#def show_image(file_path):
#  image = open_image(file_path)
#  plot_image(image)

#def show_image_array(image_array):
#  plot_image(image_array)



def fig_response(fig):
    """Turn a matplotlib Figure into Flask response"""
    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(fig, mimetype='image/png')

def nocache(response):
    """Add Cache-Control headers to disable caching a response"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response
