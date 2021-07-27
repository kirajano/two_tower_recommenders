import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers
import tensorflow_recommenders as tfrs
# Custom
import data_prep as dp
from train_test_split import train_test_split
from query_tower import CustomerModel
from candidate_tower import ProductModel
from two_tower_model import TwoTowerModel as model
from utility import get_product_name, get_product_features, visualisation


# Loading Raw Data #
attributes, interactions, popularity = dp.load_raw()

# QUERY TOWER DATA #
attributes = dp.process_attributes(attributes)
interactions = dp.process_interactions(interactions, attributes, popularity)

# CANDIDATE TOWER DATA #
products_combined = dp.process_products(popularity, attributes)


###########################
# LOAD, SPLIT AND MODELS  #
###########################

# Query Tower Data
interactions_tf = tf.data.Dataset.from_tensor_slices((interactions.to_dict("list")))

# Candidate Tower Data
products_tf = tf.data.Dataset.from_tensor_slices((products_combined.to_dict("list")))

# Train-Val-Test Split
train, validation, test = train_test_split(interactions_tf)

# Initiate towers with preprocessing and embedding layers
customer_model = CustomerModel(interactions_tf, embedding_dim=64)
product_model = ProductModel(products_tf, embedding_dim=64)


##################################
# MAIN MODEL, TRAIN AND EVALUATE #
##################################

# Get Features for Retrieval
product_features = get_product_features(products_tf)

# Main Model
model = model(customer_model, product_model, product_features)

# Training
print("**** Initializing Training ****")
print("\n")
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(train, epochs=40, validation_data=validation)
print("+++++++ Training COMPLETED +++++++")

# Visualisation
visualisation(model)

# Evaluate on the Model
train_accy = model.evaluate(train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
val_accy = model.evaluate(validation, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accy = model.evaluate(test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Train Accuracy: {train_accy:.4f}")
print(f"Test Accuracy: {test_accy:.4f}")
print(f"Validation Accuracy: {val_accy:.4f}")


#######################
# GET RECOMMENDATIONS #
#######################

# Retrieve Recommendations for top10
# Retrive Recommendations
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model, k=10)
index.index(products_tf.batch(100).map(model.candidate_model), products_tf.map(lambda x: x["CONFIG_ID"]))

# Define a Retrieval Query for Recommendations
# based on customer and actions and time
sample_customer_code = "CUST00124922"
sample_action_id = "SALE"
sample_timestamp = model.query_model.time_discrete.get_weights()[0][10]
sample_action_weight = model.query_model.customer_action_weight.get_weights()[0][1]

# Fetching the recommendations
_, product_recommendations = index({
    "CUSTOMER_CODE": np.array([sample_customer_code]),
    "ACTION_ID": np.array([sample_action_id]),
    "TIMES": np.array([sample_timestamp]),
    "WEIGHT_int": np.array([sample_action_weight])
    })

# Lookup table for convinience
product_names = get_product_name()

# With product name
print(f"Top 5 recommended products for user {sample_customer_code} with brand and series")
for prod in product_recommendations[0,:5].numpy():
  print(product_names.query("CONFIG_ID == @prod.decode(\"UTF-8\")")["PRODUCT_NAME"].values[0])








