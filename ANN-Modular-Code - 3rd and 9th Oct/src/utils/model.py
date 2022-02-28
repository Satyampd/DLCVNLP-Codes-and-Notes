from tensorflow import keras
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, no_of_classes): 

    LAYERS = [
            keras.layers.Flatten(input_shape = [28,28], name = 'inputLayer'),
            keras.layers.Dense(300, activation = 'relu'),
            keras.layers.Dense(100, activation = 'relu'),
            keras.layers.Dense(no_of_classes, activation = 'softmax')
    ]
    model = keras.models.Sequential(LAYERS)
    print(model.summary())
    model.compile(loss= LOSS_FUNCTION, optimizer = OPTIMIZER, metrics = METRICS)
    return model

def get_unique_file_name(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_file_name(filename = model_name)
    path_to_model = os.path.join(model_dir,unique_filename)
    model.save(path_to_model)    

def save_plot(data,plots_dir):
    data = pd.DataFrame(data)
    unique_filename = get_unique_file_name(filename = "training_validation.jpg")
    path_to_img = os.path.join(plots_dir,unique_filename)
    fig = data.plot()
    plt.savefig(path_to_img)