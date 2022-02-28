from utils.models import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot

import pandas as pd
def main(data, epochs, eta , model_name, plot_name):

    df_AND = pd.DataFrame(data)
    X, y = prepare_data(df_AND)

    model_AND = Perceptron(eta = ETA , epochs = EPOCHS)
    model_AND.fit(X,y)
    _ = model_AND.total_loss()

    save_model(model_AND, model_name)
    save_plot(df_AND, plot_name,  model_AND)


 
if __name__ == "__main__":
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1]
         }

    ETA = 0.3
    EPOCHS = 10     
    main(data = AND, epochs = EPOCHS, eta = ETA , model_name= "and.model", plot_name = "AND.png")