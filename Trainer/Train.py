###
#Clase  orquestadora preocupada de realizar el entrenamiento del Modelo.
#  Maneja paso a paso los módulos.
# Aplica el entrenamiento.
# Termina entregando el modelo entrenado.
###

### Decidir HiperParámetros

### Importar los datasets
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from DatasetTrainer.Trainer.DatasetLoader import DatasetOrchester
from DatasetTrainer.Trainer.ModelLoader import ModelBuilder
from DatasetTrainer.Trainer.OneCyclePolicy import LRFinder, OneCycleScheduler


def trainVGG_PV(train_url,val_url):
    nro_class = 34
    orchester = DatasetOrchester(nro_class, val_url,train_url)
    trainGenerator = orchester.getTrainDataset()
    testGenerator = orchester.getValDataset()
    train_number =  orchester.getTrainAmount()
    print(train_number)

    VGG16_Builder = ModelBuilder(trainGenerator, testGenerator, nro_class,arquitecture="VGG-16")
    VGG16_PV = VGG16_Builder.getModel()
    print(VGG16_PV.summary())

    ## Definir parametros de entrenamiento
    lr = 0
    epochs = 5
    #lr=4e-1
    if (lr == 0):
        print (" BUSCAR LR")
        return 0
    cycle_callback=get1CycleCallback(epochs, train_number, lr_max=lr)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    VGG16_PV.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ##Entrenar el Modelo
    history = VGG16_PV.fit(trainGenerator,
                        validation_data=testGenerator,
                        steps_per_epoch=100,
                        epochs=epochs,
                        validation_steps=50,
                        verbose=1)

    return VGG16_PV,history





def loadTestDataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0

    x_train = x_train[..., tf.newaxis]
    x_valid = x_valid[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)

    return train_ds, valid_ds

def testLRFinder(dataset,model):

    lr_finder = LRFinder()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(dataset,epochs=10,callbacks=[lr_finder],verbose=False)
    lr_finder.plot_loss()
    lr_finder.plot_accuracy()

def get1CycleCallback(epochs,train_number,lr_max):
    batch_size = 32
    steps = np.ceil(train_number/batch_size)*epochs
    lr_schedule = OneCycleScheduler(lr_max=lr_max,steps=steps)
    return lr_schedule



def evaluateModel(history):
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()

if __name__ == '__main__':

    train_url = "../Datasets/PlantVillage/apple"
    val_url = "../Datasets/PlantVillage/apple"

    trainVGG_PV(train_url,val_url)

