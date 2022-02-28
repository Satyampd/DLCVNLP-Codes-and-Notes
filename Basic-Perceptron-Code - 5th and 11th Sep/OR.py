from cmath import log
from pickle import TRUE
from utils.models import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok= TRUE)
logging.basicConfig(filename= os.path.join(log_dir, "running_logs.log") , level=logging.INFO, format= logging_str)

def main(data, epochs, eta , model_name, plot_name):

    logging.info(" Created OR dataframe")
    df_OR = pd.DataFrame(data)
    X, y = prepare_data(df_OR)

    model_OR = Perceptron(eta = ETA , epochs = EPOCHS)
    model_OR.fit(X,y)
    _ = model_OR.total_loss()


    save_model(model_OR, model_name)
    save_plot(df_OR, plot_name,  model_OR)


if __name__ == "__main__":
        OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1]
            }
    logging.info(">>>>> Started Training >>>>>>>")
    ETA = 0.3
    EPOCHS = 10     
    main(data = OR, epochs = EPOCHS, eta = ETA , model_name= "or.model", plot_name = "OR.png")
    logging.info("<<<<<<< Traning Completed <<<<<<<")