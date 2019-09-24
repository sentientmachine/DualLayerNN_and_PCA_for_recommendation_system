# DualLayerNN_and_PCA_for_recommendation_system


## Download the Data

MovieLens DataSet:  https://grouplens.org/datasets/movielens/

## Data has been downloaded here as:

    ratings.csv
    users.csv
    items.csv

## Reproduction of verk:

    #!/usr/bin/python
    # -*- coding: utf-8 -*-

    #Basics of Deep Learning
    #Learn a Saddle function Z as follows:
    #Z = 2X^2 - 3Y^2 + 1 + error

    #Load Libraries
    import numpy as np
    import pandas as pd

    # Visualisation
    import matplotlib.pyplot as plt
    import altair as alt

    #disable warnings for tensorflow
    import warnings
    warnings.filterwarnings('ignore')
    import tensorflow as tf
    import recoflow
    from recoflow.vis import Vis3d
    warnings.resetwarnings()

    #create some ranges of data for X and Y across 2 dimensions
    x = np.arange(start = -1, stop = 1, step = 0.01)
    y = np.arange(-1, 1, 0.01)

    #Make a meshgrid that creates a 2d plane with it.
    X, Y = np.meshgrid(x,y)
    c = np.ones([200, 200])
    e = np.random.rand(200, 200)*0.1

    #lift the plane into the Z dimension to create a saddle 3d shape
    Z = 2*X*X - 3*Y*Y + 5*c + e

    #prepare and save the image
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #save out the image
    plt.savefig('saddle.png')

## Output:


![Alt text](./saddle.png?raw=true "saddle 3d shape to be learned")



## PDF Hardcopies:

### Lesson 1:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/Day1_lesson1_keras_2_layer_alg.pdf


### Lesson 2:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/Day1_lesson2_matrix_pca_reduction.pdf

### Lesson 3:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/Day1_lesson3_matrix_reduction_and_predict.pdf






Include the colorful images from here:

https://github.com/lmcinnes/umap

