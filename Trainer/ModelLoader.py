
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from DatasetTrainer.Trainer.OneCyclePolicy import LRFinder


class ModelBuilder:

    def __init__(self, dataset_train, dataset_val, nro_classes, arquitecture="VGG-16"):

        self.dataset_train = dataset_train
        self.nro_classes = nro_classes
        self.dataset_val = dataset_val

        importer = ModelImporter()
        if arquitecture == "VGG-16":
            model = importer.getVGG16ForTransferLearning(nro_classes)

        self.model=model
        return self.model


    def getModel(self):
        return self.model

    def loadLRange(self):
        lr_finder = LRFinder()

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.dataset_train, epochs=10, callbacks=[lr_finder], verbose=False)
        lr_finder.plot_loss()
        lr_finder.plot_accuracy()

class ModelImporter:

    def __init__(self):
        pass

    ##VGG
    ## Input Shape = 224 Default
    ## top = 3 FC Layers

    #Pooling: None / Avg / max
    def loadVGGImageNet(self,pooling=None,top=False):
        model = tf.keras.applications.vgg16.VGG16(
            include_top=top,
            weights='imagenet',
            pooling = pooling
        )

        return model

    def loadVGGEmpty(self,classes):
        model = tf.keras.applications.vgg16.VGG16(
            include_top=True,
            weights=None,
            classes=classes
        )

        return model

    def loadResNetImageNet(self,pooling=None,top=False,size=224):
        model = tf.keras.applications.resnet50.ResNet50(
            input_shape=(size,size,3),
            pooling=pooling,
            include_top=top)
        return model

    def loadFrozenVGG(self):
        model = self.loadVGGImageNet()
        model.trainable = False
        return model

    def loadFrozenResNet50(self):
        model = self.loadResNetImageNet()
        model.trainable=False
        return model

    def getVGG16ForTransferLearning(self,nClasses):
        base_model = self.loadFrozenVGG()

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x=tf.keras.applications.vgg16.preprocess_input(inputs)

        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        outputs = tf.keras.layers.Dense(nClasses,activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def createVGG16TopLayers(self):
        x = tf.keras.layers.GlobalAveragePooling2D()
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)
        return x

    def getResNet50ForTransferLearning(self,nClasses):
        base_model = self.loadFrozenResNet50()
        inputs = tf.keras.Input(shape=(224, 224, 3))

        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = Dropout(0.7,seed=seed_value)(x)
        x = tf.keras.layer s.Dense(4092, activation='relu')(x)
        x = tf.keras.layers.Dense(4092, activation='relu')(x)

        outputs = tf.keras.layers.Dense(nClasses, activation='sigmoid')(x)
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