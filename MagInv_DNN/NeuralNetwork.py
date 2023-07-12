from tensorflow import keras
from functools import partial
import numpy as np
import random
from math import *


class NeuralNetwork:
    def __init__(self) -> None:
        pass

  
    def sequential_API (self,num_input, num_output, num_hiddens,num_neurons, activation_hiddens = "sigmoid",initializer= "he_normal"):

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape= [num_input]))
        keras.layers.BatchNormalization()

        # regularizerLayers = partial(keras.layers.Dense,activation = "selu",kernel_initializer= "he_normal", kernerl_regularizer =keras.regularizers.l2(0.01))
        regularizerLayers = partial(keras.layers.Dense,activation = activation_hiddens,kernel_initializer= initializer)
        # with selu activation function use "lecun_normal" initializer. With "relu" activition function use "he_normal"

        for h in range(num_hiddens):
            keras.layers.Dropout(rate=0.1)
            model.add(regularizerLayers(num_neurons))
            # keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.BatchNormalization()

        model.add(keras.layers.Dense(num_output))
        
        # s= (epochs * num_input) // 32                                      # number of steps in 20 epochs (batch size = 32)
        # lr = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=s, decay_rate=0.96, staircase=False)
        
        # optimizer = keras.optimizers.SGD(lr=1e-3)
        # model.compile(loss = "MeanSquaredError", optimizer=optimizer, metrics=["accuracy"])       

        return model


    def Wide_DNN_model(self,num_input, num_output, num_hiddens,num_neurons,epochs, activation_hiddens = "relu"):

        input_ = keras.layers.Input(shape=num_input)
        hidden1 = keras.layers.Dense(num_neurons,activation =activation_hiddens)(input_)
        hidden2 = keras.layers.Dense(num_neurons,activation=activation_hiddens)(hidden1)
        concat = keras.layers.Concatenate()([input_,hidden2])
        output =keras.layers.Dense(num_output)(concat)
        model= keras.Model(inputs=[input_], outputs=[output])
        s= (epochs * num_input) // 32                                      # number of steps in 20 epochs (batch size = 32)
        lr = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=s, decay_rate=0.96, staircase=False)
        model.compile(loss = "MeanSquaredError", optimizer=keras.optimizers.Adam(learning_rate=lr,beta_1=0.9, beta_2= 0.999), metrics=["accuracy"])       
        
        return model








# class WideAndDeepModel(keras.Model):

#     def __init__(self, units =30, num_out=24, activation= "relu", **kwargs):
#         super().__init__(**kwargs)
#         self.hidden1 = keras.layers.Dense(units, activation=activation)
#         self.hidden2 = keras.layers.Dense(units,activation=activation)
#         self.output = keras.layers.Dense(num_out)

#     def call(self,inputs):
#         hidden1 = self.hidden1(inputs)
#         hidden2 =self.hidden2(hidden1)
#         concat =keras.layers.concatenate([input, hidden2])
#         output = self.output(concat)

#         return output

