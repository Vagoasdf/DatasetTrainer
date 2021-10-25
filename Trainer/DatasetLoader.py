
import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_DATASET_URI = "../Datasets/PlantDoc-dataset"
PLANTDOC_DATASET_URI = "../Datasets/PlantDoc-dataset"
PLANTPATHOLOGY_EXTRACTED = "../Datasets/PlantPathology-extracted/train"
PLANTPATHOLOGY_ORDERED = "../Datasets/PlantPathology-order-dataset/images/train"





class DatasetOrchester:

    def __init__(self, nro_clases,target_size,batch_size=64):

        self.target_size = target_size
        self.batch_size = batch_size
        self.nro_clases = nro_clases

        self.seed = 42
        self.generator =  DatasetGenerator()

    ##setters

    def setBatchSize(self,batch):
        self.batch_size = batch

    def setTargetSize(self,target_px):
        self.target_size = target_px

    #Getters
    def getTrainDataset(self):
        return self.train_dataset

    def getTrainAmount(self):
        return  self.train_dataset.__len__

    def getValDataset(self):
        return self.val_dataset

    def getValAmount(self):
        return self.val_dataset.__len__

    ## Funciones
    def createTrainDataset(self,url):
        if(self.target_size != None):
            self.train_dataset = self.generator.loadTrainDataset(url,self.batch_size,self.target_size)
            return self.train_dataset
        self.train_dataset = self.generator.loadTrainDataset(url,self.batch_size)
        return self.train_dataset

    def createValDataset(self,url):
        if(self.target_size != None):
            self.val_dataset = self.generator.loadValidationDataset(url,self.batch_size,self.target_size)
            return self.val_dataset
        self.val_dataset = self.generator.loadValidationDataset(url,self.batch_size)
        return self.val_dataset

    def createTestDataset(self,url, batch_size,target_size=None):

        if(target_size==None):  target_size = self.target_size
        self.testDataset = self.generator.buildSimpleDataset(url,batch_size,target_size)
        return self.testDataset

    def show(self,image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy())
        plt.axis('off')


    def showTestDataset(self):
        dataset_batch = next(iter(self.train_dataset))
        self.showBatch(dataset_batch)

class DatasetGenerator:

    def __init__(self):
        self.class_names = None

    def loadTrainDataset(self,dataset_dir, batch_size, target_size=None, model_architecture="VGG"):
        if(model_architecture == "VGG"):
            if(target_size == None) :  target_size = 224
            TrainDataset = self.buildAgumentedDataset(dataset_dir, batch_size,target_size)
        else:
            print("Arquitectura no reconocida")
            pass

        return TrainDataset

    def loadValidationDataset(self, dataset_dir, batch_size, target_size = None, model_architecture="VGG"):
        if (model_architecture == "VGG"):
            if (target_size == None):  target_size = 224
            ValDataset = self.buildSimpleDataset(dataset_dir,batch_size, target_size, )
        else:
            print("Arquitectura no reconocida")
            pass

        return ValDataset

    # Crea un image generator sencillo, sin transformaciones
    def buildSimpleDataset(self, datagen_dir, batch_size,target_size):
        datagen = ImageDataGenerator(rescale=1 / 255.) # Convierte de 1-255 a 0 - 1
        dataset = datagen.flow_from_directory(
            datagen_dir,
            batch_size=batch_size,
            class_mode='categorical',
            target_size=(target_size, target_size))
        return dataset

    # Crea un image generator con transformaciones variadas.
    def buildAgumentedDataset(self, datagen_dir, batch_size,target_size):

        datagen = ImageDataGenerator(
            rescale=1. / 255, # Convierte de 1-255 a 0 - 1
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        dataset = datagen.flow_from_directory(
            datagen_dir,
            batch_size=batch_size,
            class_mode='categorical',
            target_size=(target_size, target_size))
        return dataset

