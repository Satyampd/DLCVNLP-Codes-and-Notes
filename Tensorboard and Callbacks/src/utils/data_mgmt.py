import tensorflow as tf


def get_data(validation_datasize):
    """This will fetch MNIST image data and will return train, test and validation
    Returns:
        tuples: returns (X_train, y_train),(X_valid, y_valid) (X_test, y_test)
    """
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test)=mnist.load_data()

    # Normalizing the data and dividing the training data into train and validation set/
    X_valid, X_train = X_train[:validation_datasize]/255, X_train[validation_datasize:]/255
    X_test = X_test/255 
    y_valid, y_train = y_train[:validation_datasize], y_train[validation_datasize:]

    return (X_train, y_train),(X_valid, y_valid), (X_test, y_test)