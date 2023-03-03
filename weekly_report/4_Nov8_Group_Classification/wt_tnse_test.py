import pandas as pd
import numpy as np
from PIL import Image
from peakdetect import peakdetect
import sklearn
import os

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
        # img_X = convert_img(df_['X'][interval[0]: interval[1]].values)
        # img_X = Image.fromarray(np.uint8(img_X))
        # img_X.thumbnail((60, 60))

        # img_Y = convert_img(df_['Y'][interval[0]: interval[1]].values)
        # img_Y = Image.fromarray(np.uint8(img_Y))
        # img_Y.thumbnail((60, 60))

        img_Z = convert_img(df_['Z'][interval[0]: interval[1]].values)
        img_Z = Image.fromarray(np.uint8(img_Z))
        img_Z.thumbnail((60, 60))

        # data.append(np.stack([np.array(img_X), np.array(img_Y), np.array(img_Z)], axis=2))
        data.append(np.array(img_Z).flatten())

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


# ================================   Data reading   ================================
data_path = r'/Users/hoanglinh96nl/Library/CloudStorage/OneDrive-NTHU/Projects and Papers/ASE_PHM_WireBonding/dataset/6_Oct11_data'

file_31k_p1d1 = os.path.join(data_path, 'CELL21_WBR431_31K_BY_P1D1.csv')
file_31k_p1d2 = os.path.join(data_path, 'CELL21_WBR431_31K_BY_P1D2.csv')
file_1856k_pxd1 = os.path.join(data_path, 'CELL21_WBR431_1856K_BY_PXD1.csv')
file_1856k_pxd2 = os.path.join(data_path, 'CELL21_WBR431_1856K_BY_PXD2.csv')

df_p1d1 = pd.read_csv(file_31k_p1d1, header=0).drop(['Date_Time', 'id', 'X', 'Y'], axis=1).reset_index(drop=True)
df_p1d2 = pd.read_csv(file_31k_p1d2, header=0).drop(['Date_Time', 'id', 'X', 'Y'], axis=1).reset_index(drop=True)
df_pxd1 = pd.read_csv(file_1856k_pxd1, header=0).drop(['Date_Time', 'id', 'X', 'Y'], axis=1).reset_index(drop=True)
df_pxd2 = pd.read_csv(file_1856k_pxd2, header=0).drop(['Date_Time', 'id', 'X', 'Y'], axis=1).reset_index(drop=True)

label_file = pd.read_excel(os.path.join(data_path, 'Label.xlsx'), sheet_name=0)
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

# Test data preparation
x_test = np.concatenate([x_train1, x_train2, x_train3, x_train4])
y_test_raw = np.concatenate([y_train1, y_train2, y_train3, y_train4])

# ================================   Test TSNE Algorithm   ================================
# Configure t-SNE function. 
embed = TSNE(
    n_components=2, # default=2, Dimension of the embedded space.
    perplexity=10, # default=30.0, The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    early_exaggeration=12, # default=12.0, Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. 
    learning_rate=50, # default=200.0, The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
    n_iter=1000, # default=1000, Maximum number of iterations for the optimization. Should be at least 250.
    n_iter_without_progress=100, # default=300, Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration. 
    min_grad_norm=0.00001, # default=1e-7, If the gradient norm is below this threshold, the optimization will be stopped.
    metric='euclidean', # default=’euclidean’, The metric to use when calculating distance between instances in a feature array.
    init='random', # {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default=’random’. Initialization of embedding
    verbose=1, # default=0, Verbosity level.
    random_state=42, # RandomState instance or None, default=None. Determines the random number generator. Pass an int for reproducible results across multiple function calls.
    method='barnes_hut', # default=’barnes_hut’. By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. 
    angle=0.5, # default=0.5, Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
    n_jobs=-1, # default=None, The number of parallel jobs to run for neighbors search. -1 means using all processors. 
)

# Transform X
X_embedded = embed.fit_transform(x_test)

# Print results
print('New Shape of X: ', X_embedded.shape)
print('Kullback-Leibler divergence after optimization: ', embed.kl_divergence_)
print('No. of iterations: ', embed.n_iter_)

# Create a scatter plot
fig = px.scatter(None, x=X_embedded[:,0], y=X_embedded[:,1], 
                 labels={
                     "x": "Dimension 1",
                     "y": "Dimension 2",
                 },
                 opacity=1, color=y_test_raw.astype(str))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title_text="t-SNE")
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
fig.show()