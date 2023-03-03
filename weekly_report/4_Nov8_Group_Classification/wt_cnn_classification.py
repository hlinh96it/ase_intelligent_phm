import pandas as pd
import numpy as np
from PIL import Image
from peakdetect import peakdetect
import sklearn

import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import History

from sklearn.manifold import TSNE # for t-SNE dimensionality reduction

import tensorflow as tf
tf.config.list_physical_devices()

import matplotlib.pyplot as plt
import plotly.express as px # for data visualization

plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 3)
plt.rcParams['figure.dpi'] = 150

import warnings
warnings.filterwarnings("ignore")


def peak_detection(array, look_ahead=3000, diff_threshold=0.01):
    # lookahead is the distance to look ahead from a peak to determine if it is the actual peak.
    peaks = peakdetect(array, lookahead=look_ahead)
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
    peak_lists_ = np.concatenate([higherPeaks[:, 0], lowerPeaks[:, 0]], axis=0)
    
    # improve precision of algorithm
    new_peak_lists = []
    for idx, peak in enumerate(peak_lists_):
        
        # calculate different of found peak with look_ahead data points
        different = abs(array[int(peak): int(peak) + look_ahead] - array[int(peak)])
        
        # since the complex pattern of drop down interval
        if array[int(peak) + look_ahead] - array[int(peak)] < 0:
            diff_threshold = 0.005
        
        # check for significant different between found peak and look_ahead data points
        # if there are significant different, update to new peak
        significant_diff = different[different < diff_threshold]
        new_peak_lists.append(peak + significant_diff.shape[0])
        
        diff_threshold = 0.001
    
    return new_peak_lists

def convert_img(signal):
    size = int(np.sqrt(signal.shape[0]))
    signal_ = signal[: size**2]
    signal_ = (signal_ - signal_.min()) / (signal_.max() - signal_.min())
    img = np.reshape(signal_, newshape=(size, size))
    
    return img*255

def create_dataset(df_, interval_list, label_list, train_test_ratio=0.8, test_set=False):
    data = []
    for idx_, interval in enumerate(interval_list):
        img_X = convert_img(df_['X'][interval[0]: interval[1]].values)
        img_X = Image.fromarray(np.uint8(img_X))
        img_X.thumbnail((60, 60))

        img_Y = convert_img(df_['Y'][interval[0]: interval[1]].values)
        img_Y = Image.fromarray(np.uint8(img_Y))
        img_Y.thumbnail((60, 60))

        img_Z = convert_img(df_['Z'][interval[0]: interval[1]].values)
        img_Z = Image.fromarray(np.uint8(img_Z))
        img_Z.thumbnail((60, 60))

        # data.append(np.stack([np.array(img_X), np.array(img_Y), np.array(img_Z)], axis=2))
        data.append(np.array(img_Z))

    ratio = int(len(data) * train_test_ratio)
    if test_set:
        return np.array(data), label_list.to_list()
    
    return np.array(data[: ratio]), label_list[: ratio].values, \
           np.array(data[ratio:]), label_list[ratio:].values

def crop_index(df_):
    crop_idx = []
    dary_list = []
    
    dary = df_['Piezo'].copy().rolling(window=200, min_periods=1).mean()
    dary -= np.average(dary)
    
    # using cumulative sum method
    dary_step = np.cumsum(dary)
    dary_step = (dary_step - dary_step.min()) / (dary_step.max() - dary_step.min())
    peak_lists = peak_detection(dary_step, diff_threshold=0.001)
    crop_idx.extend(np.array(peak_lists))
    dary_list.extend(dary_step)
    
    crop_idx = np.sort(np.array(crop_idx, dtype=int))
    
    crop_idx_pair = []
    for idx, val in enumerate(crop_idx):
        try:
            crop_idx_pair.append([val, crop_idx[idx + 1]])
        except IndexError:
            pass
    
    crop_idx_pair = np.array(crop_idx_pair, dtype=int)
    bonding_intervals_ = crop_idx_pair[crop_idx_pair[:, 1] - crop_idx_pair[:, 0] > 3000]
    
    return bonding_intervals_

def create_2d_cnn_model(input_shape, num_classes, x_train, y_train, x_validation, y_validation,\
    batch_size=3, epochs=5, verbose_training=1):
    history = History()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    with tf.device('/device:GPU:0'):
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose_training,
            validation_data=(x_validation, y_validation), callbacks=[history])

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = model.evaluate(x_validation, y_validation, verbose=0)
    print('Validation loss: {}, Validation accuracy: {}'.format(test_score[0], test_score[1]))

    return model


# ================================   Data reading   ================================
file_31k_p1d1 = 'CELL21_WBR431_31K_BY_P1D1.csv'
file_31k_p1d2 = 'CELL21_WBR431_31K_BY_P1D2.csv'
file_1856k_pxd1 = 'CELL21_WBR431_1856K_BY_PXD1.csv'
file_1856k_pxd2 = 'CELL21_WBR431_1856K_BY_PXD2.csv'

df_p1d1 = pd.read_csv(file_31k_p1d1, header=0).drop(['Date_Time', 'id'], axis=1).reset_index(drop=True)
df_p1d2 = pd.read_csv(file_31k_p1d2, header=0).drop(['Date_Time', 'id'], axis=1).reset_index(drop=True)
df_pxd1 = pd.read_csv(file_1856k_pxd1, header=0).drop(['Date_Time', 'id'], axis=1).reset_index(drop=True)
df_pxd2 = pd.read_csv(file_1856k_pxd2, header=0).drop(['Date_Time', 'id'], axis=1).reset_index(drop=True)

label_file = pd.read_excel('Label.xlsx', sheet_name=0)
label_file.head()

bonding_intervals_p1d1 = pd.DataFrame(crop_index(df_p1d1), columns=['p1d1_0', 'p1d1_1'])
bonding_intervals_p1d2 = pd.DataFrame(crop_index(df_p1d2), columns=['p1d2_0', 'p1d2_1'])
bonding_intervals_pxd1 = pd.DataFrame(crop_index(df_pxd1), columns=['pxd1_0', 'pxd1_1'])
bonding_intervals_pxd2 = pd.DataFrame(crop_index(df_pxd2), columns=['pxd2_0', 'pxd2_1'])

interval_label = pd.concat([bonding_intervals_p1d1, bonding_intervals_p1d2,
                            bonding_intervals_pxd1, bonding_intervals_pxd2, label_file['Group'][:-1]], axis=1)
interval_label = interval_label[interval_label['Group'].isin([5, 7, 12])]
subjects = {5:1, 7:2, 12:3}
interval_label['Group'] = interval_label['Group'].map(subjects)
interval_label_shuffle = sklearn.utils.shuffle(interval_label)  # type: ignore

labels = interval_label_shuffle['Group']


# ================================   Data preparation   ================================
x_train1, y_train1, x_validation1, y_validation1 = \
    create_dataset(df_p1d1, interval_label_shuffle[['p1d1_0', 'p1d1_1']].values, labels)
x_train2, y_train2, x_validation2, y_validation2 = \
    create_dataset(df_p1d2, interval_label_shuffle[['p1d2_0', 'p1d2_1']].values, labels)
x_train3, y_train3, x_validation3, y_validation3 = \
    create_dataset(df_pxd1, interval_label_shuffle[['pxd1_0', 'pxd1_1']].values, labels)
x_train4, y_train4, x_validation4, y_validation4 = \
    create_dataset(df_pxd2, interval_label_shuffle[['pxd2_0', 'pxd2_1']].values, labels)

x_train = np.concatenate([x_train1, x_train2, x_train3, x_train4])[: -50]
y_train = np.concatenate([y_train1, y_train2, y_train3, y_train4])[: -50]
x_validation = np.concatenate([x_validation1, x_validation2, x_validation3, x_validation4])
y_validation = np.concatenate([y_validation1, y_validation2, y_validation3, y_validation4])

# Test data preparation
x_test = np.concatenate([x_train1, x_train2, x_train3, x_train4])[-50:]
y_test_raw = np.concatenate([y_train1, y_train2, y_train3, y_train4])[-50:]


# ================================   Define model   ================================
num_classes = 4
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)

input_shape = (60, 60, 1)
batch_size = 5
epochs = 4

x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')

predict_2d_cnn_model = create_2d_cnn_model(input_shape, num_classes, x_train, y_train, x_validation, y_validation)

test = predict_2d_cnn_model.evaluate(x_test, y_test, verbose=0)
predict = predict_2d_cnn_model.predict(x_test)
result = np.argmax(predict, axis=1)
print(result)
print(y_test_raw)
print('Test loss: {}, Test accuracy: {}'.format(test[0], test[1]))
