import Train
import Evaluate
from DatasetLoader import  *
from ModelLoader import *
import tensorflow as tf

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

nro_class = 3
batch = 64
target_size = 224


def cnn_train(train_dataset, val_dataset,cnn_builder):
    cnn = cnn_builder.getModel()
    ##Hiperparametros de entrenamientos
    # determinar LR
    train_number = train_dataset.__len__()
    cnn_lr = cnn_builder.getLR(cnn)
    cnn_epochs = 20
    cnn_cycle = Train.get1CycleCallback(cnn_epochs,
                                        batch,
                                        train_number,
                                        lr_max=cnn_lr)

    # Crear checkpoints de entrenamiento
    checkpoint_path = "training_tf/cp.ckpt"
    cp_callback = Train.getCheckpointCallback(checkpoint_path)

    class_weights = Train.getClassWeights(train_dataset)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=cnn_lr)
    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)
    cnn_history = cnn.fit(train_dataset,
                          validation_data = val_dataset,
                          callbacks=[cp_callback,cnn_cycle],
                          epochs = cnn_epochs,
                          class_weight=class_weights,
                          verbose = 1,)

    return cnn, cnn_history

def cnn_finetune(modelBuilder,finetune_lr,train_dataset,val_dataset):
    epochs = 5
    finetune_optimizer = tf.keras.optimizers.RMSprop(learning_rate=finetune_lr)
    model = modelBuilder.setModelToFineTune()
    model.compile(optimizer=finetune_optimizer, loss='categorical_crossentropy', metrics=METRICS)
    class_weights = Train.getClassWeights(train_dataset)

    finetune_history = model.fit(train_dataset,
                                          validation_data=val_dataset,
                                          epochs=epochs,
                                          verbose=1,
                                          class_weight=class_weights)
    return model, finetune_history
if __name__ == '__main__':

    Train.seedRandomness()
    base_url = "../input/mixed-apples-to-apples-good-segmentations/MixedSegmented/"
    val_url = base_url+"val"
    train_url = base_url+"train"
    test_url = base_url+"test"

    ##Hiperparametros de dataset


    orchester = DatasetOrchester(nro_class,target_size,batch)
    trainDataset = orchester.createTrainDataset(train_url)
    valDataset = orchester.createValDataset(val_url)
    testDataset = orchester.createTestDataset(test_url,batch,target_size)
    vgg_builder = ModelBuilder(trainDataset,
                               nro_classes=nro_class,
                               arquitecture="VGG_16")

    vgg, vgg_history = cnn_train(trainDataset,valDataset,vgg_builder)

    resnet_builder = ModelBuilder(trainDataset,
                               nro_classes=nro_class,
                               arquitecture="RESNET-50")

    resnet, resnet_history = cnn_train(trainDataset,valDataset,resnet_builder)

    Train.saveModel(vgg,"vgg_base","transfer_output")
    Train.saveModel(resnet,"resnet_base","transfer_output")

    print("Evaluacion VGG")
    Evaluate.evaluateModel(vgg_history)

    print ("Evaluacion ResNet")
    Evaluate.evaluateModel(resnet_history)

    finetune_lr = 1e-6

    finetuned_vgg,vgg_history = cnn_finetune(vgg_builder,finetune_lr,trainDataset,valDataset)

    finetuned_resnet,resnet_history = cnn_finetune(resnet_builder,finetune_lr,trainDataset,valDataset)


    print("Evaluacion VGG")
    Evaluate.evaluateModel(vgg_history)

    print ("Evaluacion ResNet")
    Evaluate.evaluateModel(resnet_history)

    print(" Prueba de Modelos")
    Evaluate.create_cm_model(vgg,testDataset)
    Evaluate.create_cm_model(finetuned_vgg,testDataset)

    Evaluate.create_cm_model(resnet,testDataset)
    Evaluate.create_cm_model(finetuned_resnet,testDataset)

    #vgg.evaluate(testDataset)
    #finetuned_vgg.evaluate(testDataset)
    #resnet.evaluate(testDataset)
    #finetuned_resnet.evaluate(testDataset)






