import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, eta, epochs ):
        
        # below we have taken three as number because, first two would be normal X1 and X2 input 
        # and third one is for bias value
        self.weights = np.random.randn(3) * 1e-4  # multiplying small values
        self.eta = eta  # eta means learning rate
        self.epochs = epochs  # number of iterations model will have of data
        print(f"Initial weights before training: {self.weights}")
        
        
    def activationFunction(self, inputs, weights):    
        z = np.dot(inputs, weights)  # z = W*X
    #    print(f"Z:{z} and return value {np.where(z>0,1,0)}")
        return np.where(z>0,1,0)
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        # in below line, we are adding bias with every row
        # np.ones((a,b)) will return a 2d list of 1 value on 4 rows and 1 column
        X_with_bias = np.c_[self.X , -np.ones((len(self.X),1))]  
        print(f"X with Bias:\n {X_with_bias}")
        
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch: {epoch}")
            print("--"*10)
            y_hat = self.activationFunction(X_with_bias, self.weights)   #forward propagation
            print(f"Predicated value after forward pass:\n{y_hat}")
            self.error = self.y - y_hat
            print(f"Error\n{self.error}")
            self.weights = self.weights + self.eta*np.dot(X_with_bias.T, self.error)  #backward propagation
            print(f"Updated weights after epoch {epoch} -> {self.weights }")
            print("#####"*10)    
            
            
            
    def predict(self, X):
        X_with_bias = np.c_[X , -np.ones((len(X),1))] 
        return self.activationFunction(X_with_bias, self.weights)
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total Loss: {total_loss}")
        return total_loss
        