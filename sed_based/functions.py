import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import numpy as np




def reset_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and layer.kernel is not None:
            layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))





# crossvalidation function

def crossvalidate(train_val_df, model, predictors, target, kf, conv1D = False):
    '''function for crossvalidation of the model
    train_val_df - dataframe with train and validate data
    model - model to be trained
    predictors - list of predictors
    target - target variable
    kf - KFold object
    es_callback - EarlyStopping callback
    '''

   
    
    num_classes = train_val_df[target].nunique()
    print(f"Number of classes: {num_classes}")

    i=1

    acc_scores = []
    loss_scores = []

    acc_histories = []
    loss_histories = []
    val_acc_histories = []
    val_loss_histories = []

    global_confidence_scores = []


    for train_index, validate_index in kf.split(train_val_df):
        # reset weights
        reset_weights(model)
        
        train_df = train_val_df.iloc[train_index]
        validate_df = train_val_df.iloc[validate_index]

        if conv1D:
             
            X_train = train_df[predictors].values
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            y_train = keras.utils.to_categorical(train_df[target].values, num_classes=num_classes)


            X_validate = validate_df[predictors].values
            X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], 1)
            y_validate = keras.utils.to_categorical(validate_df[target].values, num_classes=num_classes)



        else:
             
            X_train = train_df[predictors].values
            y_train = keras.utils.to_categorical(train_df[target].values, num_classes=num_classes)

            X_validate = validate_df[predictors].values
            y_validate = keras.utils.to_categorical(validate_df[target].values, num_classes=num_classes)


        # fit the model
        print(f"training for {i} subset")
        history = model.fit(X_train, y_train, epochs=300, batch_size=5, validation_data=(X_validate, y_validate), verbose=0)
        # evaluate the model

        ev_results = model.evaluate(X_validate, y_validate)



        #save evaluation results 
        acc_scores.append(ev_results[1])
        loss_scores.append(ev_results[0])

        acc_histories.append(history.history['accuracy'])
        loss_histories.append(history.history['loss'])
        val_acc_histories.append(history.history['val_accuracy'])
        val_loss_histories.append(history.history['val_loss'])

        # prediction of the results 
        y_pred = model.predict(X_validate)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_validate, axis=1)
        print(f"predicted classes:       {y_pred_class}")
        print(f"true validation classes: {y_true}")

        # confidence scores 

        confidence_scores = []
        for row in y_pred:
            highest = np.max(row)
            mean = np.mean(np.delete(row, np.argmax(row)))
            confidence_score = round(((highest / mean)-1),4)
            confidence_scores.append(confidence_score)
            global_confidence_scores.append(confidence_score)
        
        mean_confidence = np.mean(confidence_scores)
        sd_confidence = np.std(confidence_scores)

        print(f"mean confidence score: {round(mean_confidence,4)}, sd confidence score: {round(sd_confidence,4)}")
        print("=====================================================")

        



        i+=1

    # calculate mean scores

    mean_acc_score = np.mean(acc_scores)
    sd_acc_score = np.std(acc_scores)
    mean_loss_score = np.mean(loss_scores)
    sd_loss_score = np.std(loss_scores)


    # part to plot mean values from history
    
    # mean and sd values for learning curves 
    mean_acc_histories = np.nanmean(acc_histories, axis=0)
    sd_acc_histories = np.nanstd(acc_histories, axis=0)
    mean_val_acc_histories = np.nanmean(val_acc_histories, axis=0)
    sd_val_acc_histories = np.nanstd(val_acc_histories, axis=0)
    mean_loss_histories = np.nanmean(loss_histories, axis=0)
    sd_loss_histories = np.nanstd(loss_histories, axis=0)
    mean_val_loss_histories = np.nanmean(val_loss_histories, axis=0)
    sd_val_loss_histories = np.nanstd(val_loss_histories, axis=0)
    

    print(f"validation set mean accuracy: {round(mean_acc_score,4)}, sd{round(sd_acc_score,4)}, mean loss: {round(mean_loss_score,4)}, sd: {round(sd_loss_score,4)}")
    print(f"global mean confidence score: {round(np.mean(global_confidence_scores),4)}, sd confidence score: {round(np.std(global_confidence_scores),4)}")
    # plotting mean curves 

    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(mean_acc_histories, label='train', color='blue')
    plt.fill_between(range(len(mean_acc_histories)), mean_acc_histories - sd_acc_histories, mean_acc_histories + sd_acc_histories, color='blue', alpha=0.2)
    plt.plot(mean_val_acc_histories, label='validate', color='orange')
    plt.fill_between(range(len(mean_val_acc_histories)), mean_val_acc_histories - sd_val_acc_histories, mean_val_acc_histories + sd_val_acc_histories, color='orange', alpha=0.2)
    plt.legend()
    plt.title("Mean Accuracy")

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(mean_loss_histories, label='train', color='blue')
    plt.fill_between(range(len(mean_loss_histories)), mean_loss_histories - sd_loss_histories, mean_loss_histories + sd_loss_histories, color='blue', alpha=0.2)
    plt.plot(mean_val_loss_histories, label='validate', color='orange')
    plt.fill_between(range(len(mean_val_loss_histories)), mean_val_loss_histories - sd_val_loss_histories, mean_val_loss_histories + sd_val_loss_histories, color='orange', alpha=0.2)
    plt.legend()
    plt.title("Mean Loss")

    plt.show()


    # calculating min and max values for loss plots range 
    flat_loss_histories = np.concatenate(loss_histories)
    flat_val_loss_histories = np.concatenate(val_loss_histories)
    min_loss = min((min(flat_loss_histories), min(flat_val_loss_histories)))
    max_loss = max((max(flat_loss_histories), max(flat_val_loss_histories)))



    # Plotting results for each fold 
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))


    # Accuracy
    for i, acc in enumerate(acc_histories):
        axs[0, 0].plot(acc, label=f'Fold {i+1}')
    axs[0, 0].set_title('Training Accuracy')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_ylim(0, 1)  # Set y-axis scale from 0 to 1
    axs[0, 0].legend()

    # Validation Accuracy
    for i, val_acc in enumerate(val_acc_histories):
        axs[0, 1].plot(val_acc, label=f'Fold {i+1}')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_ylim(0, 1)  # Set y-axis scale from 0 to 1
    axs[0, 1].legend()

    # Loss
    for i, loss in enumerate(loss_histories):
        axs[1, 0].plot(loss, label=f'Fold {i+1}')
    axs[1, 0].set_title('Training Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_ylim(min_loss,max_loss)  # Set y-axis scale based on the maximum loss value
    axs[1, 0].legend()

    # Validation Loss
    for i, val_loss in enumerate(val_loss_histories):
        axs[1, 1].plot(val_loss, label=f'Fold {i+1}')
    axs[1, 1].set_title('Validation Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].set_ylim(min_loss,max_loss)  # Set y-axis scale based on the maximum validation loss value
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()











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
    ConfusionMatrixDisplay.from_predictions(y_test_class, y_pred_class, #sample_weight=sample_weight,
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
    if "_" in x:
        num = int(x.split("_")[1])
    else:
        num = int(x)    
    
    for key, value_list in codes.items():
        if num in value_list:
            return key
    
    return None   


def pond_decoder_2(x,codes):
    
    x=x.replace('S','')
    num = int(x.split(".")[0])
    
    
    for key, value_list in codes.items():
        if num in value_list:
            return key
    
    return None   