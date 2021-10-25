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
from sklearn.utils import class_weight
from DatasetTrainer.Trainer.DatasetLoader import DatasetOrchester
from DatasetTrainer.Trainer.ModelLoader import ModelBuilder
from DatasetTrainer.Trainer.OneCyclePolicy import LRFinder, OneCycleScheduler

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


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

## CNN Train con Class Weights y CLR

def complete_cnn_train(train_dataset, val_dataset,cnn_builder,batch_size, cnn_lr=None):
    ## Base Hiperparameters
    cnn = cnn_builder.getModel()
    train_number = train_dataset.samples
    if( cnn_lr == None): cnn_lr = cnn_builder.getLR(cnn)
    cnn_epochs = 20
    ##Create 1CLR
    cnn_cycle = get1CycleCallback(cnn_epochs,
                                  batch_size,
                                  train_number,
                                  lr_max=cnn_lr)

    ##Crear CWeights
    class_weights = getClassWeights(train_dataset)

    # Crear checkpoints de entrenamiento
    checkpoint_path = "training_tf/cp.ckpt"
    cp_callback = getCheckpointCallback(checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_lr)
    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)
    cnn_history = cnn.fit(train_dataset,
                          validation_data = val_dataset,
                          callbacks=[cp_callback,cnn_cycle],
                          epochs = cnn_epochs,
                          class_weight=class_weights,
                          verbose = 1,)

    return cnn, cnn_history

##CNN train sin CLR,
def fixed_cnn_train(train_dataset, val_dataset, cnn_builder, cnn_lr=None):
    ## Base Hiperparameters
    cnn = cnn_builder.getModel()
    if( cnn_lr == None): cnn_lr = cnn_builder.getLR(cnn)
    cnn_epochs = 20

    ##Crear CWeights
    class_weights = getClassWeights(train_dataset)

    # Crear checkpoints de entrenamiento
    checkpoint_path = "training_tf/cp.ckpt"
    cp_callback = getCheckpointCallback(checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_lr)
    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)
    cnn_history = cnn.fit(train_dataset,
                          validation_data=val_dataset,
                          callbacks=[cp_callback],
                          epochs=cnn_epochs,
                          class_weight=class_weights,
                          verbose=1, )

    return cnn, cnn_history

## CNN train sin CW
def balanced_cnn_train(train_dataset, val_dataset, cnn_builder, batch_size, cnn_lr=None):
    ## Base Hiperparameters
    cnn = cnn_builder.getModel()
    train_number = train_dataset.samples
    if( cnn_lr == None): cnn_lr = cnn_builder.getLR(cnn)
    cnn_epochs = 20
    ##Create 1CLR
    cnn_cycle = get1CycleCallback(cnn_epochs,
                                  batch_size,
                                  train_number,
                                  lr_max=cnn_lr)
    # Crear checkpoints de entrenamiento
    checkpoint_path = "training_tf/cp.ckpt"
    cp_callback = getCheckpointCallback(checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_lr)
    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)
    cnn_history = cnn.fit(train_dataset,
                          validation_data = val_dataset,
                          callbacks=[cp_callback,cnn_cycle],
                          epochs = cnn_epochs,
                          verbose = 1)
    return cnn, cnn_history

## Entrenamiento básico, sin nada especial.
def basic_cnn_train(train_dataset, val_dataset, cnn_builder,cnn_lr=1e-3):
    cnn = cnn_builder.getModel()
    train_number = train_dataset.samples
    cnn_epochs = 20

    # Crear checkpoints de entrenamiento
    checkpoint_path = "training_tf/cp.ckpt"
    cp_callback = getCheckpointCallback(checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_lr)
    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)
    cnn_history = cnn.fit(train_dataset,
                          validation_data = val_dataset,
                          callbacks=[cp_callback],
                          epochs = cnn_epochs,
                          verbose = 1)
    return cnn, cnn_history

def cnn_finetune(modelBuilder,finetune_lr,train_dataset,val_dataset):
    epochs = 5
    finetune_optimizer = tf.keras.optimizers.RMSprop(learning_rate=finetune_lr)
    model = modelBuilder.setModelToFineTune()
    model.compile(optimizer=finetune_optimizer, loss='categorical_crossentropy', metrics=METRICS)
    class_weights = getClassWeights(train_dataset)

    finetune_history = model.fit(train_dataset,
                                          validation_data=val_dataset,
                                          epochs=epochs,
                                          verbose=1,
                                          class_weight=class_weights)
    return model, finetune_history


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
    classes = trainDataset.classes
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(classes),
        trainDataset.classes)
    # pasarlas a dict

    class_weights = {i: class_weights[i] for i in range(class_weights.size)}
    print(class_weights)
    return class_weights


def saveModel(model,filename,final_path):
    final_dir = os.path.dirname(final_path)
    model.save('./'+final_path+'/'+filename+'.h5')

def saveCheckpoint(model,filename,final_path):
    final_dir = os.path.dirname(final_path)
    model.save_weights('./'+final_path+'/'+filename)

