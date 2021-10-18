
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
        self.labels = None
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

    def showBatch(self,dataset_batch_tuple):
        imageBatch, labelBatch = dataset_batch_tuple
        for i in range(self.batch_size):
            image = imageBatch[i]
            label = labelBatch[i]
            self.show(image, label)

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

    """
    Intento de uso en Pipelines. No funcionó como esperaba, pero si sirve
    para cargar la base de un dataset. Deprecado porque flow_from_directory es mucho mejor
    y ya podemos recombinar y separar con Separator
    """
    def loadDatasetAsListFiles(self,dataset_url,target_size):
        dataset_dir = pathlib.Path(dataset_url)
        image_count = len(list(dataset_dir.glob('*/*.jpg')))
        dataset, labels = self.buildBaseDatasetFromlist(dataset_dir,image_count)
        dataset = dataset.map(self.process_path(dataset_dir,target_size))
        return dataset,labels

    def buildBaseDatasetFromlist(self,dataset_dir,image_count):
        list_ds = tf.data.Dataset.list_files(str(dataset_dir/'*/*'), shuffle=False)
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        for f in list_ds.take(5):
          print(f.numpy())
        self.class_names = np.array(sorted([item.name for item in dataset_dir.glob('*') if item.name != "LICENSE.txt"]))
        print(self.class_names)
        return list_ds, self.class_names

    ##Auxiliares para transformar file_path e imagenes
    def get_label(self,file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(self,img,target_size):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [target_size, target_size])

    def process_path(self,file_path,target_size):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img,target_size)
        return img, label

    def configure_batches(self,dataset,batch_size):
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        return dataset

    #Todavía por determinar la utilidad real
    def buildDatasetToTensorSlices(self,datagen_dir,nclass,target_size,batch_size):
        datagen = ImageDataGenerator(rescale=1 / 255.) # Convierte de 1-255 a 0 - 1
        datagen_flow = datagen.flow_from_directory(
            datagen_dir,
            batch_size=batch_size,
            class_mode='categorical',
            target_size=(target_size, target_size))
        #(Para referencia)
        self.inspectFlowFromDirectory(datagen_flow)
        dataset = tf.data.dataset.from_generator(
            lambda : datagen_flow,
            output_types = (tf.float32,tf.float32),
            output_shapes = ([32,target_size,target_size,3],[32,nclass])
        )

        dataset.element_spec

    def inspectFlowFromDirectory(self,datagen_flow):
        images, labels = next(datagen_flow)
        verbose = 1
        if (verbose):
            print(images.dtype, images.shape)
            print(labels.dtype, labels.shape)
        #

    def seeImagesInDataset(self,dataset):

        image_batch, label_batch = next(iter(dataset))
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            plt.title(self.class_names[label])
            plt.axis("off")
