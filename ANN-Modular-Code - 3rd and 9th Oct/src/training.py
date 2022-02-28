from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plot
import argparse
import os


def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_size"]
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
   

    loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["optimizer"]
    metrics = config["params"]["metrics"]
    no_of_classes = config["params"]["no_of_classes"]
    model = create_model(LOSS_FUNCTION=loss_function,OPTIMIZER=optimizer, METRICS=metrics, no_of_classes=no_of_classes)

    epochs = config["params"]["epochs"]
    batch_size = config["params"]["epochs"]
    model.fit(X_train, y_train, epochs = 2, validation_data= (X_valid, y_valid), batch_size = batch_size)

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

    # for reading config - this is usefull when we want to change the config file
    # from command line otherwise below is another example where we can directly give 
    # config.yaml address

    # args = argparse.ArgumentParser()
    # args.add_argument("--config", "-c", default = "config.yaml") 
    # parsed_args = args.parse_args()
    # training(config_path= parsed_args.config)
    

    # directly passing config.yaml path
    training(config_path= "config.yaml")



     