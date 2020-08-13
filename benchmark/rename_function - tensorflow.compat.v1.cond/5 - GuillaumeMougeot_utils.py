import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import tensorflow as tf

#----------------------------------------------------------------------------
# Numpy operations utils

def downscale2d_np(x, factor=2, data_format='NHWC'):
    """
    Takes a 4D array and reduces it using the factor.
    """
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return np.array(x)
    x = np.array(x)
    dtype = x.dtype
    x = x.astype(np.float32)
    y = np.zeros((x.shape[0],x.shape[1]//factor,x.shape[2]//factor,3)) if data_format=='NHWC' else np.zeros((x.shape[0],3,x.shape[2]//factor,x.shape[3]//factor))
    for i in range(factor):
        for j in range(factor):
            if data_format=='NHWC':
                y += x[:,i::factor,j::factor,:]
            else:
                y += x[:,:,i::factor,j::factor]
    return (y / (factor**2)).astype(dtype)

def upscale2d_np(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    s = x.shape
    x = np.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = np.tile(x, [1, 1, factor, 1, factor, 1])
    x = np.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
    return x

def lerp_np(a, b, t): return a + (b - a) * t

#----------------------------------------------------------------------------
# Tensorflow operations utils

def lerp(a, b, t):
    with tf.compat.v1.name_scope('Lerp'):
        return a + (b - a) * t
def cset(cur_lambda, new_cond, new_lambda):
    with tf.compat.v1.name_scope('Condition'):
        return lambda: tf.compat.v1.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Saving utils

def save_images(filename, images, image_size):
    """
    'images' are the outputs of the network.
    """
    # Output image: 256x256x3
    h,w,c = (256,256,3)
    output = np.zeros((h,w,c))

    # Resize images
    # resized_images = np.empty((len(images),32,32,3))
    # for i in range(len(resized_images)):
        # resized_images[i] = sk.transform.resize(images[i], (32,32,3), mode='constant')
    resized_images = upscale2d_np(images, 32//image_size)
    # Fill the output image
    for i in range(0,h,32):
        for j in range(0,w,32):
            output[i:i+32,j:j+32,:] = resized_images[i//32*8+j//32]

    # Figure set up: remove the axes
    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display the image
    ax.imshow(output)
    
    # Save the figure
    plt.savefig(filename, dpi = h)
    plt.close()

#----------------------------------------------------------------------------

import sys
import glob
import datetime
import pickle
import re
import numpy as np
from collections import OrderedDict 
import scipy.ndimage
import PIL.Image

#----------------------------------------------------------------------------
# Image utils.

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, format)

def save_image(image, filename, drange=[0,1], quality=95):
    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)

def save_image_grid(images, filename, drange=[0,1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)

#----------------------------------------------------------------------------
