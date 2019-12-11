import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import create_cnn

class save_result:
    def __init__(self, nn, nn_model, path_export_result):
        self.nn = nn
        self.nn_model = nn_model
        self.path_export_result = path_export_result
        
        self.save_historic()
        self.padding()
        self.save_model()
        self.padding()
        self.plot_history()

    def padding(self):
        print("=============================") 

    def save_model(self):
        self.nn_model.save(os.path.join(self.path_export_result, 'weights_model.h5'))
        print("Model H5 Saved.")

    def save_historic(self):
        with open(os.path.join(self.path_export_result, 'historic.txt'), 'wb') as file_pi:  
            pickle.dump(self.nn.history, file_pi)
        print("Model Historic Saved.")

    def plot_history(self): 
        
        plt.figure(0)
        plt.plot(self.nn.history['acc'],'r')
        plt.plot(self.nn.history['val_acc'],'g')
        plt.xticks(np.arange(0, 11, 2.0))
        plt.rcParams['figure.figsize'] = (8,6)
        plt.xlabel("Num of Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy vs Validation Accuracy")
        plt.legend(['train', 'validation'])
        plt.savefig(os.path.join(self.path_export_result, "plot_accuracy.png"))

        plt.figure(1)
        plt.plot(self.nn.history['loss'],'r')
        plt.plot(self.nn.history['val_loss'],'g')
        plt.xticks(np.arange(0, 11, 2.0))
        plt.rcParams['figure.figsize'] = (8,6)
        plt.xlabel("Num of Epchos")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Validation Loss")
        plt.legend(['train','validation'])
        plt.savefig(os.path.join(self.path_export_result, "plot_loss.png"))
        
        print("Preview Metrics-Results Saved.")

class create_model:
    def __init__(self, input_shape, n_classes, loss, optimizer, summary=True):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.loss = loss
        self.optimizer = optimizer
        self.summary = summary
    
    def prepare_nn(self):
        nn_model =  create_cnn(self.input_shape, self.n_classes, self.loss, self.optimizer, self.summary)
        
        return nn_model

def load_train_test(path):
    x_train = os.path.join(path, "x_train.npy")
    y_train = os.path.join(path, "y_train.npy")
    x_test = os.path.join(path, "x_test.npy")
    y_test = os.path.join(path, "y_test.npy")
    
    x_train = np.load(x_train)
    y_train = np.load(y_train)
    x_test = np.load(x_test)
    y_test = np.load(y_test)
    
    return x_train, y_train, x_test, y_test