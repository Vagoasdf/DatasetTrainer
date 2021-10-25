
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from DatasetTrainer.Trainer.OneCyclePolicy import LRFinder


class ModelBuilder:

    def __init__(self, dataset_train, nro_classes, arquitecture="VGG-16", seed=None):

        self.dataset_train = dataset_train
        self.nro_classes = nro_classes
        self.epochs = 0
        self.seed = seed
        self.importer = ModelImporter()
        if arquitecture == "VGG-16":
            base_model, model = self.buildVGG16()

        if arquitecture == "RESNET-50":
            base_model, model = self.buildResNet50()

        self.base_model = base_model
        self.model = model

    def buildVGG16(self):
        base_model = self.importer.loadFrozenVGG()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return base_model, model

    def buildSmallVGG16(self):
        base_model = self.importer.loadFrozenVGG()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(2048, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return base_model, model

    def buildSmallerVGG16(self):
        base_model = self.importer.loadFrozenVGG()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return base_model, model

    def buildResNet50(self):
        base_model = self.importer.loadFrozenResNet50()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = base_model(x, training=False)
        globalPooling = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dropout(0.7, seed=self.seed)(globalPooling)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(2048, activation='relu')(x)

        outputs = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return base_model, model

    def buildSmallResNet50(self):
        base_model = self.importer.loadFrozenResNet50()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = base_model(x, training=False)
        globalPooling = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dropout(0.7, seed=self.seed)(globalPooling)
        x = tf.keras.layers.Dense(2048, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        outputs = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return base_model, model

    def buildSmallerResNet50(self):
        base_model = self.importer.loadFrozenResNet50()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = base_model(x, training=False)
        globalPooling = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dropout(0.7, seed=self.seed)(globalPooling)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        outputs = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return base_model, model

    def setTrainDataset(self,trainDataset):
        self.dataset_train = trainDataset

    def getModel(self):
        return self.model

    def setModelToFineTune(self):
        # Nota:  se tiene que recompilar
        self.base_model.trainable = True
        self.model.summary()
        return self.model

    def setModelToTransferLearning(self):
        self.base_model.trainable = False
        self.model.summary()
        return self.model

    def createNewClassifier(self,nClasses):
            ##Crea un clasificador (ultima capa del modelo) con n clases
        #Caso 1: no tiene clasificador, creamos uno nuevo.
        ##Caso dos, tiene uno a siq ue le damos.
        self.model.layers.pop()
        self.nro_classes=nClasses
        outLayer = tf.keras.layers.Dense(self.nro_classes, activation='sigmoid')
        input=self.model.input
        outputs=outLayer(self.model.layers[-1].output)
        self.model = tf.keras.Model(input, outputs)
        self.model.summary()
        return self.model

    def loadLRange(self, model):
        lr_finder = LRFinder()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.dataset_train, epochs=5, callbacks=[lr_finder], verbose=False)
        lr_finder.plot_loss()
        lr_finder.plot_accuracy()
        lr_acc = lr_finder.get_best_acc_lr()
        lr_loss = lr_finder.get_best_loss_lr()

        return lr_acc, lr_loss

    def getLR(self,model):
        lr_acc, lr_loss = self.loadLRange(model)
        if (1 > lr_acc > 1e-5):
            lr = lr_acc
        elif (1e-5< lr_loss < 1):
            lr = lr_loss
        else:
            lr = 1e-3

        return lr



class ModelImporter:

    def __init__(self):
        pass

    ##VGG
    ## Input Shape = 224 Default
    ## top = 3 FC Layers

    # Top: Incluir las 3 layers primeras
    # pooling: None / avg / max
    def loadVGGImageNet(self, pooling=None, top=False):
        model = tf.keras.applications. \
            vgg16.VGG16(
            include_top=top,
            weights='imagenet',
            pooling=pooling
        )

        return model

    def loadFrozenVGG(self):
        model = self.loadVGGImageNet()
        model.trainable = False
        return model

    def loadResNet50ImageNet(self, pooling=None, top=False, size=224):
        base_model = tf.keras.applications.resnet50.ResNet50(
            input_shape=(size, size, 3),
            pooling=pooling,
            include_top=top)
        return base_model

    def loadFrozenResNet50(self, size=224):
        base_model = self.loadResNet50ImageNet()
        base_model.trainable = False
        return base_model