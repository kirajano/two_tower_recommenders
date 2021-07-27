import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np



# Defining the Candidate Tower (Product Features)

class ProductModel(tf.keras.Model):
  """ 
  Target Model representing products and its features 
  """
  def __init__(self, dataset, embedding_dim=64):
    super().__init__()

    # Passing dataset for preprocessing
    self.dataset = dataset

    ########################
    # Preprocesssing Layers#
    ########################

    # Products
    self.products_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.products_vocab.adapt(self.dataset.map(lambda x: x["CONFIG_ID"]))

    # Brand
    self.product_brand_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.product_brand_vocab.adapt(self.dataset.map(lambda x: x["BRAND"]))

    # Sales
    self.product_sales_norm = layers.experimental.preprocessing.Normalization()
    self.sales = np.concatenate(list(self.dataset.map(lambda x: x["NUMBER_OF_PRODUCT_SOLD"]).batch(1000)))
    self.product_sales_norm.adapt(self.sales)

    # Margin
    self.product_margin_norm = layers.experimental.preprocessing.Normalization()
    self.margin = np.concatenate(list(self.dataset.map(lambda x: x["GMII"]).batch(1000)))
    self.product_margin_norm.adapt(self.margin)

    # Traffic (Visits)
    self.product_traffic_norm = layers.experimental.preprocessing.Normalization()
    self.traffic = np.concatenate(list(self.dataset.map(lambda x: x["NUMBER_OF_VISIT"]).batch(1000)))
    self.product_traffic_norm.adapt(self.traffic)

    # Product Type
    self.product_type_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.product_type_vocab.adapt(self.dataset.map(lambda x: x["PRODUCT_TYPE"]))

    # Product Series
    self.product_series_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.product_series_vocab.adapt(self.dataset.map(lambda x: x["SERIES"]))

    # Product Category
    self.product_category_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.product_category_vocab.adapt(self.dataset.map(lambda x: x["PRODUCT_CATEGORY_1"]))

    # Product Gender
    self.product_gender_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.product_gender_vocab.adapt(self.dataset.map(lambda x: x["PRODUCT_GENDER"]))

    # Product Price
    self.product_price_norm = layers.experimental.preprocessing.Normalization()
    self.price_to_norm = np.append(
        [x for x in self.dataset.map(lambda x: x["PRICE"]).as_numpy_iterator()], 1.0)
    self.product_price_norm.adapt(self.price_to_norm)

    # Product Milliliters
    self.product_ml_norm = layers.experimental.preprocessing.Normalization()
    # adding 1.0 as dummy value to distribution for it to work in adapt
    self.ml_to_norm = np.append(
        [x for x in self.dataset.map(lambda x: x["ML"]).as_numpy_iterator()], 1.0)
    self.product_ml_norm.adapt(self.ml_to_norm)

    # Product Attributes
    self.product_attr_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.product_attr_vocab.adapt(self.dataset.map(lambda x: x["ATTRIBUTES"]))
    

    # Dimensions for embedding into high dimensional vectors
    self.embedding_dim = embedding_dim

    ##########################
    # Embedding + Norm Layers#
    ##########################

    self.product_embedding = models.Sequential()
    self.product_embedding.add(self.products_vocab)
    self.product_embedding.add(layers.Embedding(self.products_vocab.vocabulary_size(), self.embedding_dim))

    # Brand Embeddings
    self.brand_embedding = models.Sequential()
    self.brand_embedding.add(self.product_brand_vocab)
    self.brand_embedding.add(layers.Embedding(self.product_brand_vocab.vocabulary_size(), self.embedding_dim))

    # Sales
    self.sales = models.Sequential()
    self.sales.add(self.product_sales_norm)

    # Margin
    self.margin = models.Sequential()
    self.margin.add(self.product_margin_norm)

    # Traffic
    self.traffic = models.Sequential()
    self.traffic.add(self.product_traffic_norm)

    # Product Category
    self.category = models.Sequential()
    self.category.add(self.product_category_vocab)
    self.category.add(layers.Embedding(
              self.product_category_vocab.vocabulary_size(), self.embedding_dim))
    
    # Product Type
    self.product_type = models.Sequential()
    self.product_type.add(self.product_type_vocab)
    self.product_type.add(layers.Embedding(self.product_type_vocab.vocabulary_size(), self.embedding_dim))

    # Product Series
    self.series = models.Sequential()
    self.series.add(self.product_series_vocab)
    self.series.add(layers.Embedding(self.product_series_vocab.vocabulary_size(), self.embedding_dim))

    # Gender
    self.gender = models.Sequential()
    self.gender.add(self.product_gender_vocab)
    self.gender.add(layers.Embedding(
        self.product_gender_vocab.vocabulary_size(), self.embedding_dim))

    # Price
    self.price = models.Sequential()
    self.price.add(self.product_price_norm)

    # Milliliters
    self.milliliters = models.Sequential()
    self.milliliters.add(self.product_ml_norm)

    # Atttributes (One-Hot Encoded)
    # self.attr_onehot = models.Sequential()
    # self.attr_onehot(tf.keras.Input(shape=(143,)))

    # Product Attributes
    self.product_attr = models.Sequential()
    self.product_attr.add(self.product_series_vocab)
    self.product_attr.add(layers.Embedding(self.product_attr_vocab.vocabulary_size(), self.embedding_dim))

  def call(self, inputs):
    return tf.concat([
                      self.product_embedding(inputs["CONFIG_ID"]),
                      self.brand_embedding(inputs["BRAND"]),
                      self.sales(inputs["NUMBER_OF_PRODUCT_SOLD"]),
                      self.margin(inputs["GMII"]),
                      self.traffic(inputs["NUMBER_OF_VISIT"]),
                      self.category(inputs["PRODUCT_CATEGORY_1"]),
                      self.product_type(inputs["PRODUCT_TYPE"]),
                      self.series(inputs["SERIES"]),
                      self.gender(inputs["PRODUCT_GENDER"]),
                      self.price(inputs["PRICE"]),
                      self.milliliters(inputs["ML"]),
                      # self.attr_onehot(tf.cast(inputs["ATTRIBUTES"], dtype=tf.float32)),
                      self.product_attr(inputs["ATTRIBUTES"]),
      ], axis=1)