import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import confusion_matrix
import numpy as np
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def evaluateModel(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

        #Matriz de confusión


def create_cm_model(model,dataset):
    labels = dataset.labels
    predictions = model.predict(dataset)
    plot_cm(labels,predictions)

def plot_cm(labels, predictions):

    pred_classes = np.argmax(predictions, axis = 1)
    cm = confusion_matrix(labels, pred_classes )
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def seeConfusionMatrix(model,dataset):
    dataset_labels = dataset.labels
    dataset_predictions = model.predict(dataset)
    plot_cm(dataset_labels,dataset_predictions)