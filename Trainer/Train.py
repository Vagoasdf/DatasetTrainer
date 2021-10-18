###
#Clase  orquestadora preocupada de realizar el entrenamiento del Modelo.
#  Maneja paso a paso los módulos.
# Aplica el entrenamiento.
# Termina entregando el modelo entrenado.
# La idea es que podamos agarrar un ctrl c ctrl v y meterlo a Kaggle
#  y que funcione sin problemas
###

### Decidir HiperParámetros

### Importar los datasets
import os.path

import numpy as np
import tensorflow as tf
from DatasetTrainer.Trainer.DatasetLoader import DatasetOrchester
from DatasetTrainer.Trainer.ModelLoader import ModelBuilder
from DatasetTrainer.Trainer.OneCyclePolicy import LRFinder, OneCycleScheduler
from sklearn.utils import class_weight
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import confusion_matrix



def seedRandomness():
    #  La semilla garantiza que los números aleatorios sean símilares entre ejemplos
    # Dificil que dos ejemplos den lo mismo, pero bueno que en varias ocasiones de similar.

    seed_value = 612

    # 1. Set 'os'  hash Seed for enviroment at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    ## 5. For layers that introduce randomness like dropout, make sure to set seed values
    ##model.add(Dropout(0.25, seed=seed_value))
    0

def testLRFinder(dataset,model):

    lr_finder = LRFinder()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(dataset,epochs=10,callbacks=[lr_finder],verbose=False)
    lr_finder.plot_loss()
    lr_finder.plot_accuracy()

def get1CycleCallback(epochs, batch_size,train_number,lr_max):
    steps = np.ceil(train_number/batch_size)*epochs
    lr_schedule = OneCycleScheduler(lr_max=lr_max,steps=steps)
    return lr_schedule

def getCheckpointCallback(checkpoint_url = 'training_1/cp.ckpt'):
    os.path.dirname(checkpoint_url)
    cp_best_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_url,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
    return cp_best_callback

def getClassWeights(trainDataset):
    ## So : acá podría estar el re-Balanceador

    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(trainDataset.classes),
                trainDataset.classes)
    #pasarlas a dict
    class_weights = {i : class_weights[i] for i in range(3)}
    return class_weights

def trainVGG_PV(train_url,val_url):
    ## Set Variables##
    nro_class = 3
    epochs = 15
    batch_size = 64
    ## Set Variables##
    ##Cargar VGG builder
    orchester, nro_train = loadPVDatasets(train_url,val_url,nro_class,batch_size)
    print(nro_train)
    VGG16_Builder = loadVGGModel(orchester,nro_class)
    VGG16_PV = VGG16_Builder.getModel()
    VGG16_Builder.loadLRange()
    ##Cargar VGG Builder

    ## Definir parametros de entrenamiento
    lr = 0
    #lr=4e-1
    if (lr == 0):
        print (" BUSCAR LR")
        return 0

    cycle_callback=get1CycleCallback(epochs, batch_size, nro_train, lr_max=lr)
    best_callback = getCheckpointCallback()

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    VGG16_PV.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    trainGenerator = orchester.getTrainDataset()
    testGenerator = orchester.getValDataset()
    ##Entrenar el Modelo
    history = VGG16_PV.fit(
        trainGenerator,
        validation_data=testGenerator,
        callbacks=[cycle_callback,best_callback],
        epochs=epochs,
        verbose=1)

    return VGG16_PV,history

def loadPVDatasets(train_url,val_url,nro_class,batch_size):
    orchester = DatasetOrchester(nro_class, val_url, train_url,batch_size)
    train_number = orchester.getTrainNumber()
    return orchester, train_number

def loadVGGModel(datasetOrchester,nro_class):
    trainGenerator = datasetOrchester.getTrainDataset()
    testGenerator = datasetOrchester.getValDataset()

    VGG16_Builder = ModelBuilder(dataset_train=trainGenerator,
                                 dataset_val=testGenerator,
                                 nro_classes=nro_class,
                                 arquitecture="VGG-16")
    return VGG16_Builder

def saveModel(model,filename,final_path):
    final_dir = os.path.dirname(final_path)
    model.save('./'+final_path+'/'+filename+'.h5')

def saveCheckpoint(model,filename,final_path):
    final_dir = os.path.dirname(final_path)
    model.save_weights('./'+final_path+'/'+filename)

def loadTestDataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0

    x_train = x_train[..., tf.newaxis]
    x_valid = x_valid[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)

    return train_ds, valid_ds

