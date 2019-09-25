# DualLayerNN_and_PCA_for_recommendation_system

Dual Layer Neural Nets for Recommendation engines given limitation of small training data.

# Lesson 1:

### ipynb notebook flattened to PDF:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/Day1_lesson1_keras_2_layer_alg.pdf

# Reproduction of Lesson1 work:

Download MovieLens DataSet:  https://grouplens.org/datasets/movielens/

## Data has been downloaded in this git repo as:

    ratings.csv
    users.csv
    items.csv

## Code is reproduction of work:

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

![Alt text](./saddle.png?raw=true "saddle 3d shape to be learned")

# The above saddle is the arbitrary target Z, train the model to learn it:

    #Using Deep Learning
    #Step 0: Load Libraries

    from keras.models import Model
    from keras.layers import Dense, Input, Concatenate


    #Step 1: Design the Learning Architecture, model definition
    def deep_learning_model():

        # Get the input
        x_input = Input(shape=[1], name="X")
        y_input = Input(shape=[1], name="Y")

        # Concatenate the input
        xy_input = Concatenate(name="Concat")([x_input, y_input])

        # Create Transform functions
        Dense_1 = Dense(32, activation="relu", name="Dense1")(xy_input)
        Dense_2 = Dense(4, activation="relu", name="Dense2")(Dense_1)

        # Create the Output
        z_output = Dense(1, name="Z")(Dense_2)

        # Create the Model
        model = Model([x_input, y_input], z_output, name="Saddle")

        # Compile the Model
        model.compile(loss="mean_squared_error", optimizer="sgd")

        return model

    model = deep_learning_model()
    model.summary()


    #is this broke?
    #from keras.utils import plot_model
    #plot_model(model, show_layer_names=True, show_shapes=True)

    #Step 2 learn the weights:
    input_x = X.reshape(-1)
    input_y = Y.reshape(-1)
    output_z = Z.reshape(-1)
    X.shape, Y.shape, Z.shape

    input_x.shape, input_y.shape, output_z.shape


    df = pd.DataFrame({"X": input_x, "Y": input_y, "Z": output_z})
    df.head()

    #fit the model
    output = model.fit( [input_x, input_y], output_z, epochs=10,
                   validation_split=0.2, shuffle=True, verbose=1)

    #Step 4: Evaluate Model Performance
    from recoflow.vis import MetricsVis
    df = pd.DataFrame(output.history)
    df.reset_index()
    df["batch"] = df.index + 1
    df = df.melt("batch", var_name="name")
    df["val"] = df.name.str.startswith("val")
    df["type"] = df["val"]
    df["metrics"] = df["val"]
    df.loc[df.val == False, "type"] = "training"
    df.loc[df.val == True, "type"] = "validation"
    df.loc[df.val == False, "metrics"] = df.name
    df.loc[df.val == True, "metrics"] = df.name.str.split("val_", expand=True)[1]
    df = df.drop(["name", "val"], axis=1)
    base = alt.Chart().encode(
        x = "batch:Q",
        y = "value:Q",
        color = "type"
    ).properties(width = 300, height = 300)
    layers = base.mark_circle(size = 50).encode(tooltip = ["batch", "value"]) + base.mark_line()
    chart = layers.facet(column='metrics:N', data=df).resolve_scale(y='independent')
    chart.save('eval_model_performance.png')

![Alt text](./eval_model_performance.png?raw=true "training progress")

    #Step 5: Make a Prediction
    Z_pred = model.predict([input_x, input_y]).reshape(200,200)

    #prepare and save the image
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_pred, color='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z_pred')

    #save out the image
    plt.savefig('saddle_learned.png')


![Alt text](./saddle_learned.png?raw=true "learned saddle 3d shape")




# Tensorflow Tutorial basic Text Classification:

### ipynb notebook flattened to PDF:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/tensorflow_tutorial_basic_text_classification.pdf



### Lesson 2:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/Day1_lesson2_matrix_pca_reduction.pdf

### Lesson 3:

https://github.com/sentientmachine/DualLayerNN_and_PCA_for_recommendation_system/blob/master/Day1_lesson3_matrix_reduction_and_predict.pdf






Include the colorful images from here:

https://github.com/lmcinnes/umap

