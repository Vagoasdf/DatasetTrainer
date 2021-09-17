
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from DatasetTrainer.Trainer.OneCyclePolicy import LRFinder


class ModelBuilder:

    def __init__(self, dataset_train, dataset_val, nro_classes, arquitecture="VGG-16"):

        self.dataset_train = dataset_train
        self.nro_classes = nro_classes
        self.dataset_val = dataset_val
        self.epochs = 0
        importer = ModelImporter()
        if arquitecture == "VGG-16":
            model = importer.getVGG16ForTransferLearning(nro_classes)

        self.loadLRange(model)

        self.model = model


    def getModel(self):
        return self.model

    def loadLRange(self,model):
        lr_finder = LRFinder()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.dataset_train, epochs=10, callbacks=[lr_finder], verbose=False)
        lr_finder.plot_loss()
        lr_finder.plot_accuracy()

class ModelImporter:

    def __init__(self):
        pass

    ##VGG
    ## Input Shape = 224 Default
    ## top = 3 FC Layers

    #Top: Incluir las 3 layers primeras
    #pooling: None / avg / max
    def loadVGGImageNet(self,pooling=None,top=False):
        model = tf.keras.applications.\
            vgg16.VGG16(
            include_top=top,
            weights='imagenet',
            pooling = pooling
        )

        return model

    def loadVGGEmpty(self,classes):
        model = tf.keras.applications. \
            vgg16.VGG16(
            include_top=True,
            weights=None,
            classes=classes
        )

        return model

    def loadFrozenVGG(self):
        model = self.loadVGGImageNet()
        model.trainable = False
        return model

    def getVGG16ForTransferLearning(self,nClasses):
        base_model = self.loadFrozenVGG()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(nClasses,activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def getTestModel(self):
        return tf.keras.models.Sequential([
            Conv2D(32, 3, activation='relu'),
            MaxPool2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(10, activation='softmax')
        ])