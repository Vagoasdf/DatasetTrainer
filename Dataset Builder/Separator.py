""""
Separator;
Se encarga de separar los diferentes datasets entre sí
entregando partes segmentadas de estos datasets
Lo ideal: Que tome de un datasets N particiones de manera aleatória
Lo necesario: que divida un dataset en 2 de manera aleatoria

    Tipos de datasets considerados:
    * dir: Directorios, como las imágenes. Estandar
    * csv: Coma Separated Values. ¿Separar en 2 docs?

"""
import os
import splitfolders
import shutil

class Separator:

    def __init__(self,seed=None):
        self.aviableTypes= ["dir","csv"]
        self.currentDataset=None
        self.datasetType=None
        if(seed): self.seed = seed
        else: self.seed = 42 #placeholder

    def setImageDataset(self,dataset_url):
        self.datasetType="dir"
        self.currentDataset=dataset_url

    def splitImageDatasetWithRatio(self,output_url,ratio=(.6,.2,.2)):
        # Split with a ratio.
        if(self.currentDataset == None):
            print("No hay un dataset actualmente")
            pass
        # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
        splitfolders.ratio(self.currentDataset, output=output_url, seed=self.seed, ratio=ratio,
                           group_prefix=None)  # default values

    def splitImageDatasetByNumber(self,output_url,val_number,test_number=None,oversample=True):
        # Split val/test with a fixed number of items e.g. 100 for each set.
        if(self.currentDataset == None):
            print("No hay un dataset actualmente")
            pass
        # To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
        fixed_val = val_number
        if(test_number != None):
            fixed_val = (val_number,test_number)

        splitfolders.fixed(self.currentDataset, output=output_url, seed=self.seed, fixed=fixed_val,
                           oversample=oversample,group_prefix=None)  # default values


class Joiner:

    def __init__(self,seed=None):
        self.aviableTypes= ["dir","csv"]
        self.firstDataset=None
        self.secondDataset=None
        if(seed): self.seed = seed
        else: self.seed = 42 #placeholder

        #Unir 2 datasets con T T V (Probar, debería funcionar con ambos, y mezclado)
    def joinTwoDatasets(self,firstDataset,secondDataset,targetDir):
        # crear target dir no?
        os.makedirs(targetDir,exist_ok=True)
        self.CopyDatasetIntoTarget(firstDataset, targetDir)
        pass
        self.CopyDatasetIntoTarget(secondDataset, targetDir)

        pass

    def CopyDatasetIntoTarget(self, datasetDir, targetDir):
        first_folder = os.listdir(datasetDir)
        for dir in first_folder:
            print(dir)
            basePath = os.path.join(datasetDir, dir)
            targetPath = os.path.join(targetDir, dir)
            os.makedirs(targetPath,exist_ok=True)
            print(basePath)
            self.moveClassDirectories(basePath, targetPath)

    def moveClassDirectories(self,basePath,targetPath):
        class_folders = os.listdir(basePath)
        print (class_folders)

        for classDir in class_folders:
            baseClass=os.path.join(basePath,classDir)
            targetClass = os.path.join(targetPath,classDir)
            os.makedirs(targetClass,exist_ok=True)
            print(targetClass)
            self.moveToDirectory(baseClass,targetClass)


    def moveToDirectory(self,baseDirectory,targetDirectory):
        files = os.listdir(baseDirectory)
        for file in files:
            filePath= os.path.join(baseDirectory,file)
            shutil.move(filePath,targetDirectory)

        #Unir 2 datasets con T V
    def joinTwoBasicDatasets(self,firstDataset,secondDataset,targetDir):
        pass

        #Unir 2 datasets, T V y T T V
    def joinBasicAndCompleteDataset(self,basicDataset,CompleteDataset,targetDir):
        pass


def separatePPC(sep,targetDir,quality=None):
    basicPPCSegmented="../Datasets/Apples to Apples/CompareApplesPVPPC/PPC Segmented Apple"
    if(quality == "good"):
        basicPPCSegmented = "../Datasets/Apples to Apples/QualityApples/buena"
    elif(quality == "ok"):
        basicPPCSegmented = "../Datasets/Apples to Apples/QualityApples/media"

    sep.setImageDataset(basicPPCSegmented)
    sep.splitImageDatasetByNumber(targetDir,
                                  val_number=50,
                                  test_number=30,
                                  oversample=False),


def separatePV(sep,targetDir):
    basicPVSegmented="../Datasets/Apples to Apples/CompareApplesPVPPC/PV Segmented Apple"
    sep.setImageDataset(basicPVSegmented)
    sep.splitImageDatasetByNumber(targetDir,
                                  val_number=70,
                                  test_number=50)

def SeparateBalancedPV(sep,targetDir):
    basicPVSegmented="../Datasets/Apples to Apples/CompareApplesPVPPC/PV Balanced Apple"
    sep.setImageDataset(basicPVSegmented)
    ##Veremos que onda el oversampling. Rust tiene como 300 nomas.

    sep.splitImageDatasetByNumber(targetDir,
                                  val_number=200)


if __name__ == '__main__':
    targetPPCMed="../Datasets/PPC-Med"
    targetPPCgood="../Datasets/PPC-good"
    targetDirPV="../Datasets/PV-Segmentada"

    sep = Separator()
    #print("PPC")
    #separatePPC(sep, targetPPCgood, quality="good")
    #separatePPC(sep, targetPPCMed, quality="ok")
    #print("PV")
    #separatePV(sep,targetDirPV)
    ##Pausa para Re- Armar los train y test de cada PPC

    join = Joiner()
    targetJoinDir="../Datasets/QualitySegmentedMixed"
    join.joinTwoDatasets(targetDirPV,targetPPCgood,targetJoinDir)
    join.CopyDatasetIntoTarget(targetPPCMed,targetJoinDir)

