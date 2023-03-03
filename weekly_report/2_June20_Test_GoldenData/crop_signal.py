import pandas as pd
import glob
import os
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import ruptures as rpt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import signal
import sys

# setup matplotlib
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams['figure.dpi'] = 100


# ====================================================================
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def make_crop_index(change_location):
    # create function to crop window
    crop_window = []
    for i in range(len(change_location)):
        if i == 0:
            crop_window.append([0, change_location[i]])
            continue

        crop_window.append([change_location[i - 1], change_location[i]])

    return crop_window


def classify_signal(seq_encoder, change_loc, envelope_indices, encoder_threshold=3.0):
    """This function removes redundant interval generated from change point detection algorithm (CPA).
    Ex: CPA based on mean, variance of encoder signal to classify interval into high or low frequency, [high, high, low, high,...]
    Since there some adjacent high interval, it should belong to 1 high/low interval. This function remove that adjacent
    Resulted in [high, low, high,...]

    change_loc: list - index of interval
    """
    signal_state = []
    change_loc = [loc for loc in change_loc if loc[0]!=loc[1]]

    for idx, val in enumerate(change_loc):

        low_idx_interval = envelope_indices[(envelope_indices >= val[0]) & (envelope_indices <= val[1])]
        # print(low_idx_interval, )

        if np.mean(seq_encoder[low_idx_interval]) >= encoder_threshold:
            state = 'high'  # current state of signal as high
        else:
            state = 'low'  # current state of signal as low

        signal_state.append(state)

    return signal_state


def remove_redundant_loc(change_loc, signal_state):
    new_loc = [0]
    current_state = signal_state[0]
    signal_state_removed = [current_state]

    for idx, val in enumerate(signal_state):
        if val != current_state:
            new_loc.append(change_loc[idx][0])
            signal_state_removed.append(val)

            current_state = val

    return new_loc, make_crop_index(new_loc), signal_state_removed


def extend_envelope(df_np, env_indices_np):
    """

    :param df_np:
    :param env_indices_np:
    :return: filled missing
    """
    extend_list = []

    for idx, val in enumerate(env_indices_np):
        if idx == len(env_indices_np)-1:
            extend_list.extend([val for _ in range(len(df_np) - val)])
            return np.array(extend_list)
        if idx == 0 and env_indices_np[0] != 0:
            extend_list.extend([val for _ in range(val)])
        else:
            extend_list.extend([val for _ in range(env_indices_np[idx+1] - val)])


def plot_change_points(piezo, encoder, ts_change_loc, save_fig, directory, color='red', alpha=1):
    plt.figure(figsize=(100, 5), dpi=150)
    plt.plot(encoder)
    plt.plot(piezo)
    for x in ts_change_loc:
        plt.axvline(x, lw=2, linestyle='--', color=color, alpha=alpha)

    if save_fig:
        plt.savefig(directory + '/overall_data')

    plt.close()


def butter_highpass(cut=0.5, order=5):
    b, a = signal.butter(order, cut, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cut, order=5):
    b, a = butter_highpass(cut, order=order)
    y = signal.filtfilt(b, a, data, axis=0)
    return y


def normalize_signal(x):
    filtered_highpass = butter_highpass_filter(x, cut=0.1, order=5)
    return filtered_highpass


def sampling_data(x):
    sampled_idx = [i for i in range(0, len(x), 10)]
    return np.take(x, sampled_idx)


def compare_signal(x, preference_2, ind, directory):
    cumulative_score = 0

    # preprocessing for input signal
    x_normalized = normalize_signal(x)
    x_sampled = sampling_data(x_normalized)

    plt.plot(x_sampled)
    plt.plot(x.values)

    for i in range(len(preference_2)):
        pref_normalized = normalize_signal(preference_2[i])
        pref_sampled = sampling_data(pref_normalized)
        dist_low, path2 = fastdtw(x_sampled, pref_sampled, dist=euclidean)
        cumulative_score += dist_low

    print(f'The distance of {ind + 1} example with pref is: {cumulative_score}')
    # plt.figure(figsize=(10, 5), dpi=150)
    # if cumulative_score[0] < cumulative_score[1]:
    #     plt.plot(x.values)
    #     plt.savefig(directory + '/1st_bond_' + str(ind))
    #     plt.close()

    # else:
    #     plt.plot(x.values)
    #     plt.savefig(directory + '/2nd_bond_' + str(ind))
    #     plt.close()


# =======================================================================

# create config file 
if not os.path.exists('config.text'):
    details = {'file_name': 'April_28_data.csv',
               'piezo_column': 'Piezo',
               'encoder_column': 'Z',
               'chunk_size': 1000000,
               'rolling_size': 200,
               'sampling_rate': 50,
               'change_sensitive': 2.5,
               'threshold_piezo': 1.85,
               'threshold_outlier': 20000,
               'encoder_variance': 2,
               'max_min_encoder': 3}

    with open("config.txt", 'w') as f:
        for key, value in details.items():
            f.write('%s:%s\n' % (key, value))

params = {}
with open('config.txt', 'r') as config:
    for line in config:
        params[line.partition(':')[0]] = line.partition(':')[2][:-1].strip()

# =======================================================================

df_summary = pd.read_csv(params['file_name'], header=0, iterator=True, chunksize=int(params['chunk_size']))

change_location = []
outliers_encoder = []
outliers_piezo = []

df = pd.concat([chunk for chunk in df_summary], ignore_index=True).reset_index().drop('index', axis=1)

seq_piezo, seq_encoder = df[params['piezo_column']], df[params['encoder_column']]
indices = [i for i in range(0, len(df), int(params['sampling_rate']))]

sampling = pd.DataFrame(seq_piezo).rolling(window=int(params['rolling_size']), min_periods=1).mean()
sampling = np.take(sampling.values, indices)

# Detect the change points for encoder data
z_normalize = butter_highpass_filter(data=seq_encoder, cut=0.005, order=2)

sampling_z = np.take(z_normalize, indices)
alg_z = rpt.Pelt(model="rbf").fit(sampling_z)
change_loc_z = np.array(alg_z.predict(pen=2.5))*int(params['sampling_rate'])

# detect change point for piezo sensor data
sampling_piezo = np.take(seq_piezo.values, indices)
alg_piezo = rpt.Pelt(model="rbf").fit(sampling_piezo)
change_loc_piezo = np.array(alg_piezo.predict(pen=2.5))*int(params['sampling_rate'])

change_loc_sum = np.unique(np.sort(np.concatenate((change_loc_z, change_loc_piezo), axis=None)))
crop_window = make_crop_index(change_loc_sum)

high_envelope_indices, low_envelope_indices = hl_envelopes_idx(seq_encoder.values, dmin=50, dmax=50)
extend_low_envelope = extend_envelope(seq_encoder.values, low_envelope_indices)
signal_state = classify_signal(seq_encoder.values, crop_window, extend_low_envelope, encoder_threshold=3.0)

new_loc, new_crop, signal_state_removed = remove_redundant_loc(crop_window, signal_state)

# create folder for store data
directory = 'cropped_signal'
try:
    os.makedirs(directory, exist_ok=True)
    print("Directory '%s' created successfully" % directory)
    files = glob.glob(directory + '/*.png')
    for file in files:
        os.remove(file)
except OSError as error:
    print("Directory '%s' can not be created" % directory)

print('The cropping process is running ... ')
high_freq = {}
for idx, val in enumerate(new_crop):
    plt.figure(figsize=(16, 5), dpi=150)

    if seq_encoder[val[0]: val[1]].values.shape == (0,):
        continue

    plt.plot(seq_encoder[val[0]: val[1]])
    if len(seq_encoder[val[0]: val[1]].values) > int(params['threshold_outlier']):
        plt.savefig(directory + '/outlier_' + str(idx))
        plt.close()

    else:
        plt.savefig(directory + '/' + signal_state_removed[idx] + '_' + str(idx))
        plt.close()

    if signal_state_removed[idx] == 'high':
        high_freq[idx] = seq_encoder[val[0]: val[1]]

plot_change_points(piezo=seq_piezo, encoder=seq_encoder, ts_change_loc=new_loc, save_fig=True, directory=directory)


# ==============================  Classify high frequency data into 1st and 2nd bond looping ======
ans = input('The cropping signal process is finish! Classify 1st vs 2nd bond looping? (Y/N): ')
no = ['No', 'no', 'n', 'N']
yes = ['Yes', 'yes', 'Y', 'y']

condition = True
while condition:
    if ans in no:
        print('Thank you for using the prgram!')
        condition = False

    elif ans in yes:

        # create folder for store data
        directory = '1st_2nd_clutering'
        try:
            os.makedirs(directory, exist_ok=True)
            print("Directory '%s' created successfully" % directory)
            files = glob.glob(directory + '*.png')
            for file in files:
                os.remove(file)

        except OSError as error:
            print("Directory '%s' can not be created" % directory)

        # enter the index of 1st bond and 2nd bond looping
        num_pref = int(input('How many sample do you want to compare (enter a number): '))
        second = []

        print(
            'Since the pattern of second bond looping is similar so just enter the indices of second bond and the remaining will be classifed as first bond looping')
        for i in range(num_pref):
            second.append(int(input(f'Index of sample {i + 1} for SECOND bond: ')))

        second_pref = []
        for key in second:
            second_pref.append(high_freq[key])

        # compare each signal with their reference signal
        print('Comparing cropped signals with the reference signal ... ')
        for sig in high_freq:
            compare_signal(high_freq[sig], second_pref, sig - 1, directory=directory)

        condition = False

    else:
        ans = input('Please enter Yes or No: ')

print('Thank you for using program!')
