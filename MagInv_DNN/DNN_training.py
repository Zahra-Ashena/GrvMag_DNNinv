"""
DNN Model Training for Inversion of Magnetic Data
====================================================

This script trains Deep Neural Network (DNN) models using simulated magnetic data sets. The goal is to perform inversion of magnetic data to estimate basement topography.

Author: Zahra Ashena;
Date: July 2023

"""

# %%
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from NeuralNetwork import NeuralNetwork
import json
import matplotlib.pyplot as plt

# NN = NeuralNetwork()
# scaler = StandardScaler()
# # scaler = MinMaxScaler()
# norm = Normalizer()

# %%
# Load the forward model parameters
 
len_obs = 80.0
ndata = '10k'
nset = '1'
nexp = '1'

with open(f'N{ndata}_D{nset}.txt','r') as f:
    FM_params  = json.load(f)

y_obs = np.array(FM_params['y_obs'])
pad_x = FM_params['pad']
tot_p = FM_params['tot_p']

# %%
# Load training dataset
data = np.load(f"N{ndata}_D{nset}.npy")
nin = np.shape(y_obs)[0]                             
nout = tot_p

# %%
# Split the training set

train_set_full, test_set = train_test_split(data, test_size = 0.1, random_state = 42)
train_set, valid_set = train_test_split(train_set_full, test_size = 0.1, random_state = 42)
input_train_full = train_set_full[:,:nin]
labels_train_full = train_set_full[:,nin:]
input_test = test_set[:,:nin]
labels_test = test_set[:,nin:]

# input_train = train_set[:,:nin]
# labels_train = train_set[:,nin:]
# input_valid = valid_set[:,:nin]
# labels_valid = valid_set[:,nin:]

input_train = input_train_full
output_train = labels_train_full
# Input_trn = np.transpose(Input)
# Input_nrm = norm.fit_transform(Input)
# Input_nrm = np.transpose(Input_nrm)

input_test = input_test
output_test = labels_test
# test_trn = np.transpose(Input_test)
# test_nrm = norm.fit_transform(test_trn)
# test_trn = np.transpose(test_nrm)

# %%
# Deep Neural Network (DNN) model definition

def DNN_MLP (nhiddens,nneurons,optimizer,activation , initializer):

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape= [nin]))
    # keras.layers.BatchNormalization()
    for h in range(nhiddens):
        keras.layers.Dropout(rate=0.1)
        # regularizerLayers = partial(keras.layers.Dense,activation = activation,kernel_initializer= initializer)
        # model.add(regularizerLayers(num_neurons))
        model.add(keras.layers.Dense(nneurons, kernel_initializer=initializer,activation=activation))
        # keras.layers.BatchNormalization()

    model.add(keras.layers.Dense(nout))
    metric = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    model.compile(loss = "MeanSquaredError", optimizer=optimizer, metrics=metric)          

    return model


# %%
# DNN model hyperparameters

num_hiddens = 3
num_neurons= 300
epochs=10
batch_size = 32

""" Options for initializer"""
# initializer = keras.initializers.GlorotNormal()
# initializer = keras.initializers.HeNormal(seed=None)
initializer = keras.initializers.RandomNormal()

""" Options for activation"""
activation = keras.activations.elu

""" Options for optimizer"""
# exponential schedule
# s= epochs * len(X_train) // batch_size     # number of steps in 20 epochs (batch size = 32)
#learning_rate = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.003, decay_steps=s, decay_rate=0.1, staircase=False)
# optimizers
learning_rate=0.0001
optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
# optimizer=keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9, beta_2= 0.999)
# optimizer = keras.optimizers.SGD(learning_rate)
# power scheduling
# optimizer = keras.optimizers.SGD(lr=0.1,  decay=1e-1)

# %%
# Model training

model = DNN_MLP(num_hiddens,num_neurons,optimizer,activation = activation,initializer= initializer)
# checkpoint_cb = keras.callbacks.ModelCheckpoint("model_2Dlayers.h5",save_best_only=True)
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights =True)
# history = model.fit(X_train,y_train, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_cb,early_stopping_cb])
history = model.fit(input_train,output_train, batch_size = batch_size, epochs=epochs, validation_split=0.1)

model.save(f'N{ndata}_D{nset}_M{nexp}.h5')

# %%
# Save and Plot the training history

history_dict = history.history
with open(f'N{ndata}_D{nset}_M{nexp}_history.txt','w') as f:
    f.write(json.dumps(history_dict))  

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, epochs + 1)
plt.plot(epochs, loss_values, 'b-', label='Training loss')
plt.plot(epochs, val_loss_values, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# %%
acc_values = history_dict['mean_squared_error']
val_acc_values = history_dict['val_mean_squared_error']
plt.plot(epochs, acc_values, 'b-', label='Training')
plt.plot(epochs, val_acc_values, 'r-', label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
# plt.savefig('MSE.pdf') 
plt.show()

# %%
# Evaluate DNN model on the test set

test_mse = model.evaluate(input_test,labels_test)
print ("Test MSE: ", test_mse)

# %%
# Save the forward model parameters and DNN model hyperparameters

train_params = dict(name_data=f'rtpB{ndata}_m{nexp}.npy', num_hiddens=num_hiddens,num_neurons=num_neurons,epochs=epochs,batch_size=batch_size,
optimizer = optimizer,learning_rate=learning_rate,activation = activation,initializer= initializer)  
  
with open(f"N{ndata}_D{nset}_dnn{nexp}.txt", 'w') as f: 
    for key, value in FM_params.items(): 
        f.write('%s:%s\n' % (key, value))
        
    for key, value in train_params.items(): 
        f.write('%s:%s\n' % (key, value))


