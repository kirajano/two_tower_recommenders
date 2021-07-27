import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np


# Defining the Query Model (Customer Features)

class CustomerModel(tf.keras.Model):
  """ Features Model representing the customer and its features """
  def __init__(self, dataset, embedding_dim=64):
    super().__init__()

    # Passing dataset for preprocessing
    self.dataset = dataset

    ########################
    # Preprocesssing Layers#
    ########################

    self.customers_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.customers_vocab.adapt(self.dateset.map(lambda x: x["CUSTOMER_CODE"]))

    # Customer Interactions
    self.customer_actions = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.customer_actions.adapt(self.dateset.map(lambda x: x["ACTION_ID"]))

    # Customer Interaction Weights
    self.customer_action_weights = layers.experimental.preprocessing.IntegerLookup(mask_token=None)
    self.customer_action_weights.adapt(self.dateset.map(lambda x: x["WEIGHT_int"]))

    # Customer Continious Timestamp
    self.customer_time_norm = layers.experimental.preprocessing.Normalization()
    self.timestamps = np.concatenate(list(self.dateset.map(lambda x: x["TIMES"]).batch(1000)))
    self.customer_time_norm.adapt(self.timestamps)

    # Customer Discrete Timestamp
    days = 1100  # total days in interactions
    self.timestamps_disc = np.linspace(self.timestamps.min(), self.timestamps.max(), num=days)
    self.customer_time_disc = layers.experimental.preprocessing.Discretization(self.timestamps_disc.tolist())


    # Dimensions for embedding into high dimensional vectors
    self.embedding_dim = embedding_dim

    ##########################
    # Embedding + Norm Layers#
    ##########################
    self.customer_embedding = models.Sequential()
    self.customer_embedding.add(self.customers_vocab)
    self.customer_embedding.add(layers.Embedding(self.customers_vocab.vocabulary_size(), self.embedding_dim))


    # User Interactions via category encoding
    self.customer_action_encoding = models.Sequential()
    self.customer_action_encoding.add(self.customer_actions)
    self.customer_action_encoding.add(
        layers.Embedding(
            self.customer_actions.vocabulary_size(),
            self.embedding_dim))
    
    # Weights of user interactions
    self.customer_action_weights = models.Sequential()
    self.customer_action_weights.add(self.customer_action_weights)
    self.customer_action_weights.add(
        layers.Embedding(
            self.customer_action_weights.vocabulary_size(), self.embedding_dim)
    )

    # Time Continious
    self.time_continious = models.Sequential()
    self.time_continious.add(self.customer_time_norm)

    # Time Discrete
    self.time_discrete = models.Sequential()
    self.time_discrete.add(self.customer_time_disc)
    self.time_discrete.add(layers.Embedding(len(self.timestamps_disc) + 1, self.embedding_dim))

  # Forward pass
  def call(self, inputs):
    """ 
    Forward Pass with customer features 
    """
    return tf.concat([
                      self.customer_embedding(inputs["CUSTOMER_CODE"]),
                      self.customer_action_encoding(inputs["ACTION_ID"]),
                      self.customer_action_weight_encoding(inputs["WEIGHT_int"]),
                      self.time_continious(inputs["TIMES"]),
                      self.time_discrete(inputs["TIMES"])
    ], axis=1)