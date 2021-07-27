import data_prep as dp
import matplotlib.pyplot as plt

attributes = dp.load_raw()[0]

# Utility
# Building lookup table for product names when returning recommendations
def get_product_name():
    cols = ["CONFIG_ID", "BRAND", "SERIES"]
    product_names = attributes[cols].drop_duplicates()
    product_names["PRODUCT_NAME"] = product_names["BRAND"].str.upper() + " " + product_names["SERIES"]
    product_names.drop(["BRAND", "SERIES"], axis=1, inplace=True)
    return product_names

# Product Features
# Same params need to used in retrieval process
def get_product_features(dataset):
    product_features = dataset.map(lambda x:{
        "CONFIG_ID": x["CONFIG_ID"],
        "PRODUCT_TYPE": x["PRODUCT_TYPE"],
        "BRAND": x["BRAND"],
        "SERIES": x["SERIES"],
        "NUMBER_OF_PRODUCT_SOLD": x["NUMBER_OF_PRODUCT_SOLD"],
        "GMII": x["GMII"],
        "NUMBER_OF_VISIT": x["NUMBER_OF_VISIT"],
        "PRODUCT_CATEGORY_1": x["PRODUCT_CATEGORY_1"],
        "PRODUCT_GENDER": x["PRODUCT_GENDER"],
        "PRICE": x["PRICE"],
        "ML": x["ML"],
        "ATTRIBUTES": x["ATTRIBUTES"],
        })

    return product_features


def visualisation(model):
    
     # Plot Accuracy and Loss (Traning and Validation)
    # Accuracy
    plt.plot(model.history.history["factorized_top_k/top_100_categorical_accuracy"], label="train_accuracy")
    plt.plot(model.history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="val_accuracy")
    plt.legend()
    plt.title("Train and Validation Accuracy")
    plt.show()
    plt.close()

    # Loss
    plt.plot(model.history.history["total_loss"], label="train_loss")
    plt.plot(model.history.history["val_total_loss"], label="val_loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.title("Train and Validation Loss")
    plt.show()
    plt.close()