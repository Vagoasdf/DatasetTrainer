import Train
import Evaluate
from DatasetLoader import  *
from ModelLoader import *
import tensorflow as tf

nro_class = 3
batch = 64
target_size = 224





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

    vgg, vgg_history = Train.complete_cnn_train(trainDataset,valDataset,vgg_builder)

    resnet_builder = ModelBuilder(trainDataset,
                               nro_classes=nro_class,
                               arquitecture="RESNET-50")

    resnet, resnet_history = Train.complete_cnn_train(trainDataset,valDataset,resnet_builder)

    Train.saveModel(vgg,"vgg_base","transfer_output")
    Train.saveModel(resnet,"resnet_base","transfer_output")

    print("Evaluacion VGG")
    Evaluate.evaluateModel(vgg_history)

    print ("Evaluacion ResNet")
    Evaluate.evaluateModel(resnet_history)

    finetune_lr = 1e-6

    finetuned_vgg,vgg_history = Train.cnn_finetune(vgg_builder,finetune_lr,trainDataset,valDataset)

    finetuned_resnet,resnet_history = Train.cnn_finetune(resnet_builder,finetune_lr,trainDataset,valDataset)


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






