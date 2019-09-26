#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

#import recoflow
import numpy as np
import pandas as pd
from recoflow.preprocessing import EncodeUserItem, StratifiedSplit

import matplotlib.image as mpimg
import PIL
import matplotlib.pyplot as plt

warnings.resetwarnings()
ratings = pd.read_csv("ratings.csv")
users = pd.read_csv("users.csv")
items = pd.read_csv("items.csv")

#Cheeky guys trying to hide their secret encoding sauce behind recoflow?  Not on my watch.
from sklearn.preprocessing import LabelEncoder
def EncodeUserItem(df, user_col, item_col, rating_col, time_col):
    """
    Function to encode users and items

    Params:
        df (pd.DataFrame): Pandas data frame to be used.
        user_col (string): Name of the user column.
        item_col (string): Name of the item column.
        rating_col (string): Name of the rating column.
        timestamp_col (string): Name of the timestamp column.

    Returns:
        encoded_df (pd.DataFrame): Modifed dataframe with the users and items index
        n_users (int): number of users
        n_items (int): number of items
        user_encoder (sklearn.LabelEncoder): Encoder for users.
        item_encoder (sklearn.LabelEncoder): Encoder for items.
    """

    interaction = df.copy()

    user_encoder = LabelEncoder()
    user_encoder.fit(interaction[user_col].values)
    n_users = len(user_encoder.classes_)

    item_encoder = LabelEncoder()
    item_encoder.fit(interaction[item_col].values)
    n_items = len(item_encoder.classes_)

    interaction["USER"] = user_encoder.transform(interaction[user_col])
    interaction["ITEM"] = item_encoder.transform(interaction[item_col])

    interaction.rename({rating_col: "RATING", time_col: "TIMESTAMP"}, axis=1, inplace=True)

    print("Number of users: ", n_users)
    print("Number of items: ", n_items)

    return interaction, n_users, n_items, user_encoder, item_encoder


# Encoding the data
interaction, n_users, n_items, user_encoder, item_encoder = EncodeUserItem(ratings,
                                                                          "user_id",
                                                                          "movie_id",
                                                                          "rating",
                                                                          "unix_timestamp")


train, test = StratifiedSplit(interaction, [0.8, 0.2])
train.shape, test.shape
#((80000, 7), (20000, 7))
min_rating = interaction.RATING.min()
max_rating = interaction.RATING.max()
min_rating, max_rating
#(1, 5)

# Model - Deep Factorisation
from keras.models import Model
from keras.layers import Embedding, Input, Dot, Add, Activation, Lambda, Concatenate, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam


def DeepMF(n_users, n_items, n_factors, min_rating, max_rating):

    # Item Layer
    item_input = Input(shape=[1], name='Item')
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer="glorot_normal",
                               name='ItemEmbedding')(item_input)
    item_vec = Flatten(name='FlattenItemE')(item_embedding)

    # Item Bias
    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer="glorot_normal",
                          name='ItemBias')(item_input)
    item_bias_vec = Flatten(name='FlattenItemBiasE')(item_bias)

    # User Layer
    user_input = Input(shape=[1], name='User')
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6),
                              embeddings_initializer="glorot_normal",
                               name='UserEmbedding')(user_input)
    user_vec = Flatten(name='FlattenUserE')(user_embedding)

    # User Bias
    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-6),
                              embeddings_initializer="glorot_normal",
                              name='UserBias')(user_input)
    user_bias_vec = Flatten(name='FlattenUserBiasE')(user_bias)

    # Concatenation of Item and User & then Add Bias
    Concat = Concatenate(name="Concat")([item_vec, user_vec])
    ConcatDrop = Dropout(0.5)(Concat)

    # Use the Dense layer for non-linear interaction learning
    Dense_1 = Dense(32, name="Dense1", activation="relu")(ConcatDrop)
    Dense_1_Drop = Dropout(0.5)(Dense_1)
    Dense_2 = Dense(1, name="Dense2")(Dense_1_Drop)

    # Add the Bias
    AddBias = Add(name="AddBias")([Dense_2, item_bias_vec, user_bias_vec])

    # Scaling for each user
    y = Activation('sigmoid')(AddBias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)

    # Model Creation
    model = Model([user_input, item_input], rating_output, name="DeepFM")

    # Compile Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

    return model



n_factors = 40
model = DeepMF(n_users, n_items, n_factors, min_rating, max_rating)

from keras.utils import plot_model
plot_model(model, show_layer_names=True, show_shapes=True, to_file='lesson4_model.png')

model.summary()


output = model.fit([train.USER, train.ITEM], train.RATING,
                  batch_size=32, epochs=5, verbose=1,
                  validation_data=([test.USER, test.ITEM], test.RATING))


from recoflow.vis import MetricsVis
MetricsVis(output.history)


from recoflow.recommend import GetPredictions

#%%time
predictions = GetPredictions(model, interaction)
#CPU times: user 33.2 s, sys: 4.01 s, total: 37.3 s
#Wall time: 23.7 s
predictions.shape
#(1586126, 3)
from recoflow.metrics import RatingMetrics
RatingMetrics(test, predictions)

from recoflow.recommend import ItemEmbedding, UserEmbedding
item_embedding = ItemEmbedding(model, "ItemEmbedding")
user_embedding = UserEmbedding(model, "UserEmbedding")
#Getting Recommendation

#Given an Item, what are the similiar items?
#Given a User, what items should I recommend?
from recoflow.recommend import GetSimilar, ShowSimilarItems
similar_items = GetSimilar(item_embedding, k=5, metric="euclidean")
similar_items

#ShowSimilarItems(1, similar_items, item_encoder, items, image_path='/tf/notebooks/data/data/posters/')
item_index = 1
item_similar_indices = similar_items
image_path='./posters/'

movie_title = items.iloc[0].title
s = item_similar_indices[item_index]
movie_ids = item_encoder.inverse_transform(s)
images = []
titles = []
for movie_id in movie_ids:
    img_path = image_path + str(movie_id) + '.jpg'
    images.append(mpimg.imread(img_path))
    title = items[items.movie_id == movie_id].title.tolist()[0]
    titles.append(title)

plt.figure(figsize=(20,10))
columns = 6
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title(titles[i])

plt.savefig('lesson4_movie_titles.png')


from recoflow.recommend import GetRankingTopK
#Ranking Top K

#Get all the predictions for all users
#Remove all the items the user has already seen (train)
#Sort the remaining data by predicted ratings
#Cut-off at K
#%%time
ranking_topk = GetRankingTopK(model, interaction, train, k=5)
#CPU times: user 36.2 s, sys: 4.13 s, total: 40.3 s
#Wall time: 26.8 s
n_users * 5
#4715
ranking_topk.shape
#(4715, 4)
ranking_topk.head()


from recoflow.models import NeuralCollaborativeFiltering
ncf = NeuralCollaborativeFiltering(n_users, n_items, n_factors, min_rating, max_rating)
plot_model(ncf, show_layer_names=True, show_shapes=True)







print("done")
