import numpy as np
import pandas as pd

####################
# Loading Raw Data #
####################

def load_raw():
    attributes = pd.read_csv("CSV/attributes.csv")
    interactions = pd.read_csv("CSV/interactions_customers_products.csv")
    popularity = pd.read_csv("CSV/popularity.csv")

    return attributes, interactions, popularity


####################
# QUERY TOWER DATA #
####################

def process_attributes(attributes) -> pd.DataFrame:
    """
    Preprocessing Attributes before merging with data for candidate and query towers
    """

    # Excluding product_id and leaving only config_id as identifier
    attributes = attributes.loc[: ,attributes.columns != "PRODUCT_ID"].drop_duplicates("CONFIG_ID")

    # Putting all one-hot encoded feature columns in one concat column and hash the values
    # (to be used as a vector)
    attribute_cols = attributes.columns[8:].to_list()
    one_hot_encoded_features = attributes[attribute_cols].values.tolist()
    attr_hashed = [str(hash(tuple(attr))) for attr in one_hot_encoded_features]
    attributes["ATTRIBUTES"] = attr_hashed

    return attributes


def process_interactions(interactions, attributes, popularity, time_split=False) -> pd.DataFrame:
    """
    Preprocess Customer Interactions as basis for query tower
    """
    
    # Time-wise split
    if time_split:
        # Sorting by timestamp for later train-val-test split
        interactions["TIMES"] = pd.to_datetime(interactions["TIMES"])
        interactions.sort_values(by=["TIMES"], ascending=True, inplace=True)
        interactions.reset_index(drop=True, inplace=True)

    # Formatting timestamps for tensorflow
    interactions["TIMES"] = pd.to_numeric(pd.to_datetime(interactions["TIMES"])).astype("float32")

    # Map Popularity

    interactions = pd.merge(interactions, popularity,
                            left_on="CONFIG_ID", right_on="CONFIG_ID", how="left")
    # Cast added cols to float32 --> import into tf
    pop_cols = ["NUMBER_OF_PRODUCT_SOLD", "GMII", "NUMBER_OF_VISIT"]
    interactions[pop_cols] = interactions[pop_cols].astype("float32")
    # Remove NAs (products without sales, visits, profit info) --> used in Normalization
    interactions.dropna(subset=pop_cols, inplace=True)

    # Map attributes

    # Excluding one-hot encoded ones for now
    attr_cols = ['CONFIG_ID', 'PRODUCT_CATEGORY_1', 'PRODUCT_TYPE',
        'BRAND', 'SERIES', 'PRODUCT_GENDER', 'PRICE', 'ML', "ATTRIBUTES"]
    interactions = pd.merge(interactions, 
            attributes[attr_cols], 
            left_on="CONFIG_ID", right_on="CONFIG_ID", how="left")


    # Weights as mean per customer
    interactions["WEIGHT_cont"] = interactions.groupby("CUSTOMER_CODE").agg({"WEIGHT": np.mean})
    # Cast as int
    interactions["WEIGHT_int"] = interactions["WEIGHT"] * 10
    interactions["WEIGHT_int"] = interactions["WEIGHT_int"].astype(int)

    return interactions

########################
# CANDIDATE TOWER DATA #
########################

def process_products(popularity, attributes) -> pd.DataFrame:
    """
    Preprocess data for candidate tower
    """
    
    # Excluding one-hot encoded ones for now
    attr_cols = ['CONFIG_ID', 'PRODUCT_CATEGORY_1', 'PRODUCT_TYPE',
        'BRAND', 'SERIES', 'PRODUCT_GENDER', 'PRICE', 'ML', "ATTRIBUTES"]

    # Getting Unique products and their features --> needed for Retrieval to be deduplicated
    products_combined =  pd.merge(popularity, attributes[attr_cols],
                left_on="CONFIG_ID", right_on="CONFIG_ID", how="inner")

    return products_combined



