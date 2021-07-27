import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import tensorflow_recommenders as tfrs


# Setting up Two Tower Architecture

class TwoTowerModel(tfrs.models.Model):
  """ Retrieval model encompassing customers as query and products as candidates"""

  def __init__(self, customer_model, product_model, product_features):
    super().__init__()
    self.cutomer_model = customer_model
    self.product_model = product_model
    self.product_features = product_features

    # Query tower
    self.query_model = models.Sequential()
    self.query_model.add(self.customer_model)
    self.query_model.add(layers.Dense(64, activation="relu"))
    self.query_model.add(layers.Dense(32, activation="relu"))
    self.query_model.add(layers.Dense(32))

    # Candidate tower
    self.candidate_model = models.Sequential()
    self.candidate_model.add(self.product_model)
    self.candidate_model.add(layers.Dense(64, activation="relu"))
    self.candidate_model.add(layers.Dense(32, activation="relu"))
    self.candidate_model.add(layers.Dense(32))

    # Retrieval task for loss function
    metrics = tfrs.metrics.FactorizedTopK(
            candidates=product_features.batch(128).map(self.candidate_model)
    )
    self.task = tfrs.tasks.Retrieval(metrics=metrics)
  
  def compute_loss(self, features, training=False):
    # Passing the embeddings into the loss function
    customer_embeddings = self.query_model({
                                    "CUSTOMER_CODE": features["CUSTOMER_CODE"],
                                    "ACTION_ID": features["ACTION_ID"],
                                    "WEIGHT_int": features["WEIGHT_int"],
                                    "TIMES": features["TIMES"]
                                    })
    
    product_embeddings = self.candidate_model(
        {"CONFIG_ID": features["CONFIG_ID"], 
         "BRAND": features["BRAND"], 
         "NUMBER_OF_PRODUCT_SOLD": features["NUMBER_OF_PRODUCT_SOLD"],
         "GMII": features["GMII"],
         "NUMBER_OF_VISIT": features["NUMBER_OF_VISIT"],
         "PRODUCT_CATEGORY_1": features["PRODUCT_CATEGORY_1"], 
         "PRODUCT_TYPE": features["PRODUCT_TYPE"],
         "SERIES": features["SERIES"],
         "PRODUCT_GENDER": features["PRODUCT_GENDER"],
         "PRICE": features["PRICE"],
         "ML": features["ML"],
         "ATTRIBUTES": features["ATTRIBUTES"],
         })

    # Calculate the loss via task for query and candidate embeddings
    return self.task(customer_embeddings, product_embeddings)