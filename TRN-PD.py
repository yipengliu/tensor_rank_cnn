import keras
from keras.models import Sequential, load_model
from keras.layers import Conv3D, AveragePooling3D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import scipy.io as scio
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import os

path = 'TRN-PD'

isExists=os.path.exists(path)
if not isExists:
    os.makedirs(path)

log_filepath = path
filepath_checkpoint = 'TRN-PD_data/model.h5'

epochs = 60
batch_size = 32

CPD_data = scio.loadmat('TRN-PD_data/CPD_noise_110.mat')
x_train = CPD_data['CPD_data']
x_train = x_train.reshape(2000, 50, 110, 3, 1)
rank_data = scio.loadmat('TRN-PD_data/rank_data.mat')
y_train = rank_data['rank_data']
test_CPD_data = scio.loadmat('TRN-PD_data/val_CPD_noise_110.mat')
x_test = test_CPD_data['CPD_data']
x_test = x_test.reshape(100, 50, 110, 3, 1)
test_rank_data = scio.loadmat('TRN-PD_data/val_rank_data.mat')
y_test = test_rank_data['rank_data']


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 35, 45, 55 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 55:
        lr *= 1e-3
    elif epoch > 45:
        lr *= 1e-2
    elif epoch > 35:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


model = Sequential()

model.add(Conv3D(32, (3, 3, 2), padding='same',
                input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv3D(32, (3, 3, 2), padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 1)))

model.add(Conv3D(64, (3, 3, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(64, (3, 3, 2), padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 1)))

model.add(Conv3D(64, (3, 3, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(64, (3, 3, 2), padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 3)))

# the model so far outputs 4D feature maps (height, width, features)
model.add(Flatten())  # this converts our 4D feature maps to 1D feature vectors
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

adam = optimizers.Adam(lr=lr_schedule(0))
model.compile(loss='mae',
            optimizer=adam,
            metrics=['mae'])

y_train = (y_train - 50)/50
y_test = (y_test - 50)/50

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=0)

callbacks = [lr_reducer, lr_scheduler, tb_cb]

model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            shuffle=True)


# Load a trained model
# model = load_model(filepath_checkpoint)

y_test_pre = model.predict(x_test)
y_test_pre = np.int8(np.round(y_test_pre*50))
print('The MAE of rank prediction is', np.sum(np.absolute((y_test*50 - y_test_pre))) / 100)

