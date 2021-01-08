#!/usr/bin/env python
# coding: utf-8

# # Color Distribution Model To Cartoonise Images
# Rudraksh Kapil

# imports
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from colory.color import Color  # package for converting hex to name of color
from sklearn.cluster import KMeans
from tqdm import tqdm


# plot settings:
SMALL_SIZE = 14
MEDIUM_SIZE = 16
LARGE_SIZE = 20

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

# gets and modifies a single image given its path
def get_image(path, WIDTH, HEIGHT):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)                  # get image in RGB format
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA) # reduce size
    image = image.reshape(-1,3)                                                # flatten height and width
    return image

# converts color from RGB to hex 
def RGB2HEX(color):
    ''' Function to convert color from RGB to hex '''
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# returns a list of strings corresponding to the list of given hex_colors
def hex_list_to_names(hex_colors):
    out = [] # to be returned
    
    # go through each hex_color h
    for h in hex_colors:
        h_str = str(h)                  # convert to string for function call
        col = Color(h_str, 'wiki')      # call Color() to get object that has name
        to_append = col.name + f"\n({h})" # we want name + hex in label
        out.append(to_append)           # append to out
            
    return out

#### main function
def cartoonise_image(config):
    WIDTH = config.width
    HEIGHT = config.height

    # get image from given path
    image = get_image(config.path, WIDTH, HEIGHT) 
    orig_image = image.copy()   # save original for comparison

    # make and run K-means classifier 
    clf = KMeans(n_clusters = config.num_colors)
    labels = clf.fit_predict(image)

    # get count of each label and sort according to label 
    counts = Counter(labels)
    counts = dict(sorted(counts.items())) 
    
    # get mean colors - note order
    center_colors = clf.cluster_centers_                       # jumbled up
    ordered_colors = [center_colors[i] for i in counts.keys()] # sorted
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()] # RGB

    # cartoonise by setting each pixel to have its mean's colour
    image[:,:] = clf.cluster_centers_[labels[:]]
    

    ### plot original image
    plt.figure(figsize = (18,6), dpi = 120)
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.title("Original Image")
    plt.imshow(orig_image.reshape(HEIGHT,WIDTH,3))

    ### plot pie chart
    plt.subplot(1,3,2)
    plt.title("Color Distribution Chart")
    labels = hex_list_to_names(hex_colors)
    _,_,autotexts = plt.pie(counts.values(), labels = labels, colors = hex_colors, autopct='%1.0f%%', pctdistance=0.8)
    for t in autotexts:
        t.set_color('white')

    ### plot cartoonised image
    plt.subplot(1,3,3)
    plt.title(f"Cartoonized - {config.num_colors} colors")
    plt.axis('off')
    plt.imshow(image.reshape(HEIGHT, WIDTH, 3)) 
    plt.imsave('cartoonised_image.png', image.reshape(HEIGHT, WIDTH, 3))

    plt.savefig("combined.png")

    ### save the distribution alone
    plt.clf()
    plt.figure(figsize=(10,10), dpi=100)
    plt.title("Color Distribution Chart")
    labels = hex_list_to_names(hex_colors)
    _,_,autotexts = plt.pie(counts.values(), labels = labels, colors = hex_colors, autopct='%1.0f%%', pctdistance=0.8)
    for t in autotexts:
        t.set_color('white')
    plt.savefig("color_distribution.png")


# run the code:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_colors', type = int, default = 5)
    parser.add_argument('--path', type = str, default = 'images/1.png')
    parser.add_argument('--width', type = int, default = 1200)
    parser.add_argument('--height', type = int, default = 800)

    config = parser.parse_args()
    print(config)
    cartoonise_image(config)

    print("Images saved in enclosing directory. Thank you for using this program!")





