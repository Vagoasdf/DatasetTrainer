
import os
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_DATASET_URI = "../Datasets/PlantDoc-dataset"
PLANTDOC_DATASET_URI = "../Datasets/PlantDoc-dataset"
PLANTPATHOLOGY_EXTRACTED = "../Datasets/PlantPathology-extracted/train"
PLANTPATHOLOGY_ORDERED = "../Datasets/PlantPathology-order-dataset/images/train"


class DatasetOrchester:

    def __init__(self, nro_clases,url_validacion,url_entrenamiento):
        self.nro_clases = nro_clases
        generator =  DatasetGenerator()
        self.train_dataset = generator.loadTrainDataset( dataset_dir=url_entrenamiento)
        self.val_dataset = generator.loadValidationDataset( dataset_dir= url_validacion)



        #TODO:   Validar que tengan la misma cantidad de clases
        #self.nro_train = DatasetGenerator.getTrainDatasetCount()
        #self.nro_val = DatasetGenerator.getTrainDatasetCount()

    #Getters
    def getTrainDataset(self):
        return self.train_dataset

    def getTrainAmount(self):

        return  self.train_dataset.__len__

    def getValDataset(self):
        return self.val_dataset

    def getValAmount(self):
        return self.nro_val



class DatasetGenerator:

    def loadTrainDataset(self,dataset_dir,model_architecture="VGG"):
        if(model_architecture == "VGG"):
            target_size=224
            batch_size=20
            TrainDataset = self.buildAgumentedDataset(dataset_dir,target_size,batch_size)
        else:
            print("Arquitectura no reconocida")
            pass

        return TrainDataset

    def loadValidationDataset(self, dataset_dir, model_architecture="VGG"):
        if (model_architecture == "VGG"):
            target_size = 224
            batch_size = 20
            ValDataset = self.buildSimpleDataset(dataset_dir, target_size, batch_size)
        else:
            print("Arquitectura no reconocida")
            pass

        return ValDataset

    # Crea un image generator sencillo, sin transformaciones
    def buildSimpleDataset(self, datagen_dir, target_size, batch_size):
        datagen = ImageDataGenerator(rescale=1 / 255.) # Convierte de 1-255 a 0 - 1
        dataset = datagen.flow_from_directory(
            datagen_dir,
            batch_size=batch_size,
            class_mode='categorical',
            target_size=(target_size, target_size))
        return dataset

    # Crea un image generator con transformaciones variadas.
    def buildAgumentedDataset(self, datagen_dir, target_size, batch_size):

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


    ## TODO. ¿que onda class names?
    def showTrainImages(self,dataset,nImages):
        class_names=["AAAAA"]
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(nImages):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")






