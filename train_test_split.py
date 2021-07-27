import tensorflow as tf

def train_test_split(dataset):
    # Train 70% - Validation 10% Test 20%
    size = dataset.cardinality().numpy()

    # Shuffle
    # REMOVE FOR TIMEWISE SPLITTING
    tf.random.set_seed(42)
    shuffled = dataset.shuffle(size, seed=42, reshuffle_each_iteration=False)

    # Split
    train_size = round(size * 0.7)
    val_size = round((size - train_size) * 0.1)
    test_size = size - train_size - val_size

    # Shuffle train again
    # ADJUST FOR TIMEWISE SPLITTING
    train = shuffled.take(train_size).shuffle(train_size).batch(8192).cache()
    validation = shuffled.skip(train_size).take(val_size).batch(4096).cache()
    test = shuffled.skip(train_size + val_size).take(test_size).batch(4096).cache()

    return train, validation, test