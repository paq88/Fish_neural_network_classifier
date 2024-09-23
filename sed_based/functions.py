import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
# Function for plotting probability for each class
def plot_prob(y_pred):
    """Plots the highest probability out of the output vector of a model for each class."""
    prob_value = list(map(max, y_pred))
    y_pred_class = y_pred.argmax(axis=1)

    predict_df = pd.DataFrame({"pred_class": y_pred_class, "prob_value": prob_value})
    classes = np.sort(predict_df["pred_class"].unique())
    
    num_classes = len(classes)
    nrow = int(np.ceil(num_classes/3))
    ncol = int(np.ceil(num_classes/nrow))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4*num_classes, 3.5))

   
    for i, clas in enumerate(classes):
        ax = axes[i]
        ax.hist(predict_df.loc[predict_df["pred_class"] == clas, "prob_value"], bins=10, orientation='horizontal')
        ax.set_title(f"Probability distribution for predicted class {clas}")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Frequency")

    plt.tight_layout()
    plt.show()
    # function for evaluatig model 

def plot_prob_density(y_pred):
        """Plots probablity density plot out of the output vector of a model for each class.
        """
        prob_value = list(map(max, y_pred))
        y_pred_class = y_pred.argmax(axis=1)

        predict_df = pd.DataFrame({"pred_class": y_pred_class, "prob_value": prob_value})
        classes = np.sort(predict_df["pred_class"].unique())
        
        num_classes = len(classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for clas in classes:
            sns.kdeplot(predict_df.loc[predict_df["pred_class"] == clas, "prob_value"], label=f"Class {clas}", ax=ax, bw_adjust=0.5)
        
        ax.set_title("Probability Density for Predicted Classes")
        ax.set_ylabel("Density")
        ax.set_xlabel("Values")
        ax.set_xlim(0, 1)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

def evaluate_model(model, history, X_test, y_test, acc_name = "Accuracy", loss_name = "Loss"):
    print(model.evaluate(X_test, y_test))
    y_pred = model.predict(X_test)
    y_pred_class = y_pred.argmax(axis=1)
    y_test_class = y_test.argmax(axis=1)
    print("Predicted values:", y_pred_class)
    print("True values:", y_test_class)
    # Assuming plot_prob is defined elsewhere
    #plot_prob(y_pred)

    sample_weight = (y_pred_class != y_test_class)
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_pred_class, y_test_class, #sample_weight=sample_weight,
                                             normalize="true", values_format=".0%")

    # Plot training and validation loss
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], 'r', marker='.', label="Train Loss")
    ax.plot(history.history["val_loss"], 'b', marker='.', label="Validation loss")
    ax.legend()
    ax.set_title(loss_name)

    # Plot training and validation accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"], 'r', marker='.', label="Train accuracy")
    ax.plot(history.history["val_accuracy"], 'b', marker='.', label="Validation accuracy")
    ax.legend()
    ax.set_title(acc_name)


#creting decoder for 5 classes (original experimental setups)
# codes = {0 : [8,14,18,24,29], 1: [12,16,19,23,26], 2 : [10,21,28,30,32], 3 : [9,13,17,20,27], 4 : [11,15,22,25,31]}

def pond_decoder(x,codes):
    
    x=x.replace('S','')
    num = int(x.split("_")[1])
    
    
    for key, value_list in codes.items():
        if num in value_list:
            return key
    
    return None   
#pond_decoder("gut_14")


