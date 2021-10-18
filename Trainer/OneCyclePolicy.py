import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import Callback

class LRFinder(Callback):
    ##Callback que va ajustando el Learning Rate desde un mo hasta
    ## llegar a un maximo, ploteandose Loss y LR para encontrar el
    ## Parametró ideal de minimos y maximos de LR
    ## Codigo adapto desde: https://www.avanwyk.com/finding-a-learning-rate-in-tensorflow-2/
    ## para mas info, leer Superconvergencia de Leslie Smith.

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 5, max_steps: int = 100, smoothing=0.9):
        self.start_lr, self.end_lr = start_lr,end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        ##Inicializamos variables para guardar datos
        self.step,  self.lr = 0, 0
        self.best_acc, self.best_loss, self.avg_loss, self.avg_acc = 0, 0, 0, 0
        self.lrs, self.losses, self.accuracies = [], [], []

    ##Al comenzar el entrenamiento nos aseguramos que las variables que guarden datos se inicializen en 0
    def on_train_begin(self, logs=None):
        self.step,  self.lr = 0, 0
        self.best_acc, self.best_loss, self.avg_loss, self.avg_acc = 0, 0, 0, 0
        self.lrs, self.losses, self.accuracies = [], [], []

    ## Al comenzar el batch, ponemos el nuevo LR
    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr,self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        step = self.step
        ##Registramos Loss
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)

            if step == 0 or loss < self.best_loss:
                self.best_loss = smooth_loss
        ##Registramos Accuracy
        if acc:
            self.avg_acc = self.smoothing * self.avg_acc + (1 - self.smoothing) * acc
            smooth_acc = self.avg_acc / (1 - self.smoothing ** (self.step + 1))
            self.accuracies.append(smooth_acc)

            if step == 0 or acc < self.best_acc:
                self.best_acc = smooth_acc

        self.lrs.append(self.lr)

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    ##Incrementamos LR exponencialmente
    def exp_annealing(self,step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot_loss(self):
        ig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        plt.show()

    ##Deprecada. No encuentro Accuracy en Keras.
    def plot_accuracy(self):
        ig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.accuracies)
        plt.show()

    def get_best_acc_lr(self):

        print("Best acc:" + str(self.best_acc))
        best_index = self.accuracies.index(self.best_acc)
        best_lr = self.lrs[best_index]
        print(best_lr)
        return best_lr

    def get_best_loss_lr(self):
        print("Best Loss: " + str(self.best_loss))
        best_index = self.losses.index(self.best_loss)
        best_lr = self.lrs[best_index]
        print(best_lr)
        return best_lr

##Ciclo basado en Cosine en vez de step directo
##basado en : https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/
class CosineAnnealer:

    def __init__(self,start,end,steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n  +=1
        ##Determinar el coseno de el momento actual
        cos = np.cos(np.pi * (self.n / self.steps)) +1
        ## para dar el angulo necesario del LR de este momento
        return self.end + (self.start - self.end) / 2. * cos

class OneCycleScheduler(Callback):
    """ Callback that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
        If the model supports a momentum parameter, it will also be adapted by the schedule.
        The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
        only two phases are used and the adaptation is done using cosine annealing.
        In phase 1 the LR increases from lrmax÷fac→r to lrmax and momentum decreases from mommax to mommin
        In the second phase the LR decreases from lrmax to lrmax÷fac→r⋅1e4 and momemtum from mommax to mommin
        By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter phase1_pct
    """

    def __init__(self, lr_max, steps,
                 mom_min=0.85, mom_max=0.95,
                 phase_1_pct=0.4, div_factor=4.):
        ##Por alguna razon en avanwyk div_factor era 25. Lo decidi cambiar a 4 en base al paper
        #queda acá en caso de que sea necesario
        #div_factor = 25.

        ##Calculo de LR  Min, Max, y Fases
        lr_min = lr_max / div_factor
        final_lr = lr_min / 1e4
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        print("LR: Max= ")
        print(lr_max)
        print("LR: Min = ")
        print(lr_min)
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        ##Phases: Guarda los valores de LR y Momentum en cada fase
        ## Utiliza el alineador de Cosenos.
        self.phases = [
            [CosineAnnealer(lr_min, lr_max, phase_1_steps),
             CosineAnnealer(mom_max, mom_min, phase_1_steps)],
            [CosineAnnealer(lr_max, final_lr, phase_2_steps),
             CosineAnnealer(mom_min, mom_max, phase_2_steps)]
            ]
        ##Para plottear LR y Momentum al final.
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0
        ## Setear LR y Momentum
        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.get_lr()
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1 #Pasamos a la siguiente fase

        #Siguiente LR y momentum porfavor
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def on_train_end(self, logs=None):
        self.plot()
        pass

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self,lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass #Ignoramos

    def set_momentum(self,mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum,mom)
        except AttributeError:
            pass #ignoramos

    def lr_schedule(self):
        #Nota: self.phase == 0 || Primera fase
        #Nota: self.phase == 1 || Segunda fase
        return self.phases[self.phase][0]

    def mom_schedule(self):
        #Nota: self.phase == 0 || Primera fase
        #Nota: self.phase == 1 || Segunda fase
        return self.phases[self.phase][1]

    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
        plt.show()
