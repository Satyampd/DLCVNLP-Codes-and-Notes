from gc import callbacks
from utils.common import read_config, get_log_path, get_unique_file_name
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plot
import os
from tensorflow import keras

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_size"]
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
   

    loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["optimizer"]
    metrics = config["params"]["metrics"]
    no_of_classes = config["params"]["no_of_classes"]
    model = create_model(LOSS_FUNCTION=loss_function,OPTIMIZER=optimizer, METRICS=metrics, no_of_classes=no_of_classes)

    # creating tb log path
    unique_log_dir_name = get_log_path()
    log_dir = config["logs"]["tensorboard_logs_dir"]
    log_dir_path = os.path.join(log_dir,unique_log_dir_name)
    os.makedirs(log_dir_path, exist_ok=True)

    # tensorboard callback
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir = log_dir_path)
    # Early stopping 
    early_stop_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights= True)
    # Checkpoint callbacks
    ckpt_file_name = 'model.{epoch:02d}-{val_loss:.2f}.h5'

    ckpt_model_dir = config["artifacts"]["ckpt_model_dir"] 
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    ckpt_path = os.path.join(artifacts_dir,ckpt_model_dir)
    os.makedirs(ckpt_path, exist_ok= True)
    ckpt_path_dir = os.path.join(ckpt_path,ckpt_file_name)
    checkpoint_cb = keras.callbacks.ModelCheckpoint( ckpt_path_dir, save_best_only= True)
    # list of callbacks 
    callback_list = [tensorboard_cb, early_stop_cb, checkpoint_cb]


    epochs = config["params"]["epochs"]
    batch_size = config["params"]["epochs"]
    model.fit(X_train, y_train, epochs = 5, validation_data= (X_valid, y_valid), batch_size = batch_size, callbacks = callback_list)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["params"]["model_name"]
    save_model(model=model, model_name=model_name, model_dir= model_dir_path)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    plot_dir = config["artifacts"]["plots_dir"]
    plots_dir_path = os.path.join(artifacts_dir,plot_dir)
    os.makedirs(plots_dir_path, exist_ok=True)
    save_plot(data=model.history.history, plots_dir= plots_dir_path )



if __name__ == '__main__':

    # directly passing config.yaml path
    training(config_path= "config.yaml")



     