import pandas as pd
import glob
import os
import argparse
import numpy as np
import ruptures as rpt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import signal
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import warnings
warnings.filterwarnings("ignore")

plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['agg.path.chunksize'] = 100000


def hl_envelopes_idx(s, dmin, dmax, split=False):
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


def classify_signal(seq_encoder, piezo, change_loc, low_envelope_indices, high_envelope_indices):
    method = str(params['high_low_method'])
    threshold_piezo = float(params['threshold_piezo'])
    threshold_encoder = float(params['threshold_encoder'])
    signal_state = []
    change_loc = [loc for loc in change_loc if loc[0] != loc[1]]

    for idx, val in enumerate(change_loc):
        piezo_signal = piezo[val[0]: val[1]]
        encoder_signal = seq_encoder[val[0]: val[1]]
        low_idx_interval = low_envelope_indices[(low_envelope_indices >= val[0]) & (low_envelope_indices <= val[1])]
        high_idx_interval = high_envelope_indices[(high_envelope_indices >= val[0]) & (high_envelope_indices <= val[1])]
        if method == 'RLK':
            if np.mean(piezo_signal) <= threshold_piezo:
                state = 'low'  # current state of signal as low
            else:
                if len(low_idx_interval) == 0 or len(high_idx_interval) == 0:
                    if max(encoder_signal) - min(encoder_signal) > threshold_encoder:
                        state = 'high'  # current state of signal as high
                    else:
                        state = 'low'  # current state of signal as low
                else:
                    if ((np.mean(seq_encoder[high_idx_interval]) - np.mean(
                            seq_encoder[low_idx_interval])) > threshold_encoder):
                        state = 'high'  # current state of signal as high
                    else:
                        state = 'low'  # current state of signal as low
        elif method == 'UMT':
            if np.mean(piezo_signal) <= threshold_piezo:
                state = 'low'  # current state of signal as low
            else:
                if ((np.mean(seq_encoder[high_idx_interval]) - np.mean(
                        seq_encoder[low_idx_interval])) > threshold_encoder):
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
        if idx == 0: continue
        if val != current_state:
            new_loc.append(change_loc[idx][0])
            signal_state_removed.append(current_state)
        if idx == (len(signal_state) - 1):
            new_loc.append(change_loc[idx][1])
            signal_state_removed.append(val)

        current_state = val
    return new_loc, make_crop_index(new_loc), signal_state_removed


def remove_close_crop(change_loc, signal_state):
    new_loc = [0]
    signal_state_removed = [signal_state[0]]
    close_threshold = int(params['threshold_outlier']) / 20
    for idx, val in enumerate(change_loc):
        if idx == 0: continue
        if (change_loc[idx] - new_loc[-1]) > close_threshold:
            new_loc.append(change_loc[idx])
            signal_state_removed.append(signal_state[idx])
    return new_loc, make_crop_index(new_loc), signal_state_removed


def extend_envelope(df_np, env_indices_np):
    extend_list = []

    for idx, val in enumerate(env_indices_np):
        if idx == len(env_indices_np) - 1:
            extend_list.extend([val for _ in range(len(df_np) - val)])
            return np.array(extend_list)
        if idx == 0 and env_indices_np[0] != 0:
            extend_list.extend([val for _ in range(val)])
            extend_list.extend([val for _ in range(env_indices_np[idx + 1] - val)])
        else:
            extend_list.extend([val for _ in range(env_indices_np[idx + 1] - val)])


def plot_change_points(signal, ts_change_loc, save_fig, filename, channel, directory):
    # encoder.index = range(skiprows_num,skiprows_num+nrows_num)
    # ts_change_loc=np.array([x+skiprows_num for x in ts_change_loc])
    # change_loc_piezo=np.array([x+skiprows_num for x in change_loc_piezo])
    # change_loc_z=np.array([x+skiprows_num for x in change_loc_z])

    plt.figure(figsize=(100, 5), dpi=150)
    for ch in channel:
        if ch == 'piezo':
            break
        plt.plot(signal[ch_col[ch]], label=f'{ch} encoder')
    plt.plot(signal[ch_col['piezo']], label='piezo')
    for x in ts_change_loc:
        # plt.axvline(x, lw=2, linestyle='--', color=color, alpha=alpha)
        if (x == change_loc_piezo_id).any():
            plt.axvline(x, lw=2, linestyle='--', color='red', alpha=1)
            continue
        elif (x == change_loc_z_id).any():
            plt.axvline(x, lw=2, linestyle='--', color='black', alpha=1)
    plt.plot([], [], label='change location piezo', linestyle='--', color='red')
    plt.plot([], [], label='change location z encoder', linestyle='--', color='black')
    plt.legend(loc="upper right")
    y_locator = MultipleLocator(0.5)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_locator)
    plt.ylim(-1, 4)
    if save_fig:
        plt.savefig(directory + '/' + filename)
    plt.close()


def plot_max_min(piezo, encoder, ts_change_loc, low_envelope_indices_id, high_envelope_indices_id, file_name,
                 directory):
    plt.figure(figsize=(100, 5), dpi=150)
    plt.plot(encoder, label='encoder')
    plt.plot(piezo, label='piezo')
    for x in ts_change_loc:
        # plt.axvline(x, lw=2, linestyle='--', color=color, alpha=alpha)
        if (x == change_loc_piezo_id).any():
            plt.axvline(x, lw=2, linestyle='--', color='red', alpha=1)
            continue
        elif (x == change_loc_z_id).any():
            plt.axvline(x, lw=2, linestyle='--', color='black', alpha=1)
    plt.scatter(low_envelope_indices_id,  # x軸資料
                encoder[low_envelope_indices_id],  # y軸資料
                c="black",  # 點顏色
                s=50,  # 點大小)
                )
    plt.scatter(high_envelope_indices_id,  # x軸資料
                encoder[high_envelope_indices_id],  # y軸資料
                c="green",  # 點顏色
                s=50,  # 點大小) )
                )
    plt.plot([], [], label='change location piezo', linestyle='--', color='red')
    plt.plot([], [], label='change location z encoder', linestyle='--', color='black')
    y_locator = MultipleLocator(0.5)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_locator)
    plt.ylim(-1, 4)
    plt.legend(loc="upper right")
    plt.savefig(directory + '/' + file_name)
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
    # sampled_idx = [i for i in range(extend_envelope, len(x), 10)]
    sampled_idx = [i for i in range(0, len(x), 10)]
    return np.take(x, sampled_idx)


def compare_signal(x, preference_2, ind):
    cumulative_score = 0

    # preprocessing for input signal
    x_normalized = normalize_signal(x)
    x_sampled = sampling_data(x_normalized)

    for i in range(len(preference_2)):
        pref_normalized = normalize_signal(preference_2[i])
        pref_sampled = sampling_data(pref_normalized)
        dist_low, path2 = fastdtw(x_sampled, pref_sampled, dist=euclidean)
        cumulative_score += dist_low

    return cumulative_score


# =====================Plot high and low frequency pictures==============
def plot_high_low(encoder_channel):
    for ch in encoder_channel:
        directory = 'cropped_signal_' + str(ch)
        try:
            os.makedirs(directory, exist_ok=True)
            print("Directory '%s' created successfully" % directory)
            files = glob.glob(directory + '/' + '*.png')
            for file in files:
                os.remove(file)
        except OSError as error:
            print("Directory '%s' can not be created" % directory)
        plot_change_points(signal=df, ts_change_loc=new_loc_id, \
                           save_fig=True, filename='overall.png', channel=ch, directory=directory)
        plot_change_points(signal=df, ts_change_loc=change_loc_sum_id, \
                           save_fig=True, filename='original change points.png', channel=ch, directory=directory)
        for idx, val in enumerate(new_crop_id):
            plt.figure(figsize=(16, 5), dpi=150)
            y_locator = MultipleLocator(0.5)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_locator)
            plt.ylim(-1, 4)
            # if df[ch_col[ch]].loc[val[0]: val[1]].values.shape == (0,):
            if val[0] == val[1]:
                continue
            plt.plot(df[ch_col[ch]].loc[val[0]: val[1]])
            if len(df[ch_col[ch]].loc[val[0]: val[1]].values) > int(params['threshold_outlier']):
                plt.savefig(directory + str(idx) + '_outlier')
            else:
                plt.savefig(directory + '/' + str(idx) + '_' + signal_state_removed[idx])
            plt.close()

