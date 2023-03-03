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

start = time.time()


# ====================================================================
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
                           save_fig=True, filename='origianl change points.png', channel=ch, directory=directory)
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


# ====================================================================

parser = argparse.ArgumentParser(
    prog='crop_signal.py',
    description="Python program to automatic dividing encoder/piezo signals."
)

parser.add_argument('input_data',
                    type=str,
                    help='path to ".csv" file')

# parser.add_argument('skiprows_num',
#                     nargs='?',
#                     default=1,
#                     type=int,
#                     help='skiprows')
#
# parser.add_argument('nrows_num',
#                     nargs='?',
#                     type=int,
#                     help='nrows')
parser.add_argument('configure_file',
                    nargs='?',
                    type=str,
                    default='configure.txt',
                    help='path to "configure" file')

input_path = parser.parse_args().input_data
skiprows_num = 1
configure_file = parser.parse_args().configure_file
# chunk_size = parser.parse_args().chunk_size
# threshold_piezo = parser.parse_args().threshold_piezo

# =======================================================================
# =======================================================================

# create config file 
if not os.path.exists('configure.txt'):
    details = {'chunk_size': 1000000,
               'rolling_size': 100,
               'sampling_rate_piezo': 50,
               'sampling_rate_z': 50,
               'change_sensitive_piezo': 0.5,
               'change_sensitive_z': 2,
               'dmin': 80,
               'dmax': 80,
               'threshold_piezo': 1.85,
               'threshold_encoder': 2,
               'threshold_outlier': 10000,
               'high_low_method': 'RLK'}

    with open("configure.txt", 'w') as f:
        for key, value in details.items():
            f.write('%s:%s\n' % (key, value))

params = {}
with open(configure_file, 'r') as config:
    for line in config:
        params[line.partition(':')[0]] = line.partition(':')[2][:-1].strip()

# =======================================================================
change_location = []
encoder_channel = []
second = []
second_pref = []
high_freq = {}
low_freq = {}
outlier = {}
dtw_score = {}
# ====================Read file for crop============from========================
df_summary = pd.read_csv(input_path, header=None, iterator=True, chunksize=int(params['chunk_size']), \
                         skiprows=skiprows_num)
df = pd.concat([chunk for chunk in df_summary], ignore_index=True).reset_index().drop('index', axis=1)
print(f'data length: {len(df)}')
print(df.head())
print(df.tail())
# ====================Assign channel column=============================================
print('Please count the columns starting from 0')
ch_col = {}
ch_signal = {}
ch_col['piezo'] = int(input(f'The column of Piezo: '))
for ch in ['x', 'y', 'z']:
    ch_col[ch] = int(input(f'The column of {ch}_encoder (-1 if no need to crop) :'))  # Define x_col,y_col,z_col
    if ch_col[ch] != -1:
        encoder_channel.append(ch)

    # ====================================================================
ch_signal['piezo'] = df[ch_col['piezo']]

for ch in encoder_channel:
    ch_signal[ch] = df[ch_col[ch]]

indices_z = [i for i in range(0, 0 + len(ch_signal['piezo']), int(params['sampling_rate_z']))]
indices_piezo = [i for i in range(0, 0 + len(ch_signal['piezo']), int(params['sampling_rate_piezo']))]

# Detect the change points for encoder data
z_normalize = butter_highpass_filter(data=ch_signal['z'], cut=0.005, order=2)
sampling_z = np.take(z_normalize, indices_z)
alg_z = rpt.Pelt(model="rbf").fit(sampling_z)
# alg_z = rpt.BottomUp(model="rbf").fit(sampling_z)


change_loc_z = np.array(alg_z.predict(pen=float(params['change_sensitive_z']))) * int(params['sampling_rate_z'])

# detect change point for piezo sensor data
sampling_piezo = pd.DataFrame(ch_signal['piezo']).rolling(window=int(params['rolling_size']), min_periods=1).mean()
sampling_piezo = np.take(sampling_piezo.values, indices_piezo)

alg_piezo = rpt.BottomUp(model="clinear").fit(sampling_piezo)
change_loc_piezo = np.array(alg_piezo.predict(pen=float(params['change_sensitive_piezo']))) * int(
    params['sampling_rate_piezo'])

change_loc_sum = np.unique(np.sort(np.concatenate((change_loc_z, change_loc_piezo), axis=None)))
crop_window = make_crop_index(change_loc_sum)

low_envelope_indices, high_envelope_indices = hl_envelopes_idx(ch_signal['z'].values \
                                                               , dmin=int(params['dmin']), dmax=int(params['dmax']))

extend_low_envelope = extend_envelope(ch_signal['z'].values, low_envelope_indices)

extend_high_envelope = extend_envelope(ch_signal['z'].values, high_envelope_indices)
signal_state = classify_signal(ch_signal['z'].values, ch_signal['piezo'].values, crop_window, extend_low_envelope,
                               extend_high_envelope)

method = str(params['high_low_method'])
if method == 'RLK':
    new_loc_a, new_crop_a, signal_state_removed_a = remove_redundant_loc(crop_window, signal_state)
    new_loc_b, new_crop_b, signal_state_removed_b = remove_close_crop(new_loc_a, signal_state_removed_a)
    new_loc, new_crop, signal_state_removed = remove_redundant_loc(new_crop_b, signal_state_removed_b)
elif method == 'UMT':
    new_loc_a, new_crop_a, signal_state_removed_a = remove_redundant_loc(crop_window, signal_state)
    new_loc_b, new_crop_b, signal_state_removed_b = new_loc_a, new_crop_a, signal_state_removed_a
    new_loc, new_crop, signal_state_removed = new_loc_a, new_crop_a, signal_state_removed_a

n = len(new_crop)
i = 0
while i < n:
    if i == 0:
        i = i + 1
        pass
    elif new_crop[i] == [0, 0]:
        new_crop.pop(i), new_loc.pop(i), signal_state_removed.pop(i)
        n = n - 1
    else:
        i = i + 1

# =====================Reindex of data and change points==================
df.index = range(skiprows_num, skiprows_num + len(ch_signal['piezo']))
new_crop_id = np.array(new_crop)
new_crop_id = new_crop_id + skiprows_num
new_loc_id = np.array([x + skiprows_num for x in new_loc])
change_loc_piezo_id = np.array([x + skiprows_num for x in change_loc_piezo])
change_loc_z_id = np.array([x + skiprows_num for x in change_loc_z])
change_loc_sum_id = np.array([x + skiprows_num for x in change_loc_sum])
# =====================Reindex of data for plot==================
# new_loc_0a_id=np.array([x+skiprows_num for x in new_loc_0a])
new_loc_a_id = np.array([x + skiprows_num for x in new_loc_a])
low_envelope_indices_id = np.array([x + skiprows_num for x in high_envelope_indices])
high_envelope_indices_id = np.array([x + skiprows_num for x in low_envelope_indices])

# =====================Determine high and low frequency==================

for idx, val in enumerate(new_crop_id):
    if val[0] == val[1]:
        continue
    if signal_state_removed[idx] == 'high' and len(df[ch_col['piezo']].loc[val[0]: val[1]].values) <= int(
            params['threshold_outlier']):
        high_freq[idx] = [val[0], val[1]]
    elif signal_state_removed[idx] == 'low' and len(df[ch_col['piezo']].loc[val[0]: val[1]].values) <= int(
            params['threshold_outlier']):
        low_freq[idx] = [val[0], val[1]]
    elif len(df[ch_col['piezo']].loc[val[0]: val[1]].values) > int(params['threshold_outlier']):
        outlier[idx] = [val[0], val[1]]

# =================Piezo===================
process_class_method = 'piezo'
if process_class_method == 'piezo':
    prev_low = ''
    for sig in low_freq:
        if np.mean(df[ch_col['piezo']].loc[low_freq[sig][0]:low_freq[sig][1]]) < float(params['threshold_piezo']):
            current_low = '1st bond'
            low_freq[sig].append(current_low)
            if sig - 1 in high_freq:
                if prev_low == current_low:
                    high_freq[sig - 1].append('high freq others')
                else:
                    high_freq[sig - 1].append('reset motion')
            if sig + 2 not in low_freq:
                if sig + 1 in high_freq:
                    high_freq[sig + 1].append('looping')
        else:
            current_low = '2nd bond'
            low_freq[sig].append(current_low)
            if sig - 1 in high_freq:
                if prev_low == current_low:
                    high_freq[sig - 1].append('high freq others')
                else:
                    high_freq[sig - 1].append('looping')
            if sig + 2 not in low_freq:
                if sig + 1 in high_freq:
                    high_freq[sig + 1].append('reset motion')
        prev_low = current_low
    for sig in high_freq:
        if len(high_freq[sig]) == 2:
            high_freq[sig].append('high freq others')
for sig in outlier:
    outlier[sig].append('outlier')
overall_freq = {**high_freq, **low_freq, **outlier}

# =====================Save classified results in raw_directory==================
# create folder
# raw_directory = 'clutering_raw_data'
# try:
#     os.makedirs(raw_directory, exist_ok=True)
#     print("Directory '%s' created successfully" % raw_directory)
#     files = glob.glob(raw_directory + '/' + '*')
#     for file in files:
#         os.remove(file)
# except OSError as error:
#     print("Directory '%s' can not be created" % raw_directory)

# for sig in overall_freq:
#     # print(f'The {sig} crop is identified as {overall_freq[sig][2]}')
#     df.loc[overall_freq[sig][0]:overall_freq[sig][1], :]. \
#         to_csv(raw_directory + '/' + str(sig) + '_' + str(overall_freq[sig][2]) + '.csv', index=False, header=False)

# =====================For validation==================

overall_keys = sorted(overall_freq.keys())
window_max = overall_keys[-1]

overall_list = [[key] + overall_freq[key] for key in overall_keys]

sig = 0
wire_num = 0
while sig <= window_max - 1:

    if overall_list[sig][3] == '1st bond':
        wire_num += 1
    elif overall_list[sig][3] == 'outlier':
        wire_num += 1
    overall_list[sig].append(f'wire {wire_num}')
    sig += 1

# =====================Create validation table==============
wire_max = wire_num
order = ['1st bond', 'looping', '2nd bond', 'reset motion']
# DTW_window={'looping':[],'2nd bond':[],'reset motion':[]}
valid_df = pd.DataFrame(index=range(wire_max + 1),
                        columns=['wire num', '1st bond start', '1st bond end', 'looping start', 'looping end' \
                            , '2nd bond start', '2nd bond end', 'reset motion start', 'reset motion end'])

sig = 0
wire_num = 0
valid_df['wire num'].iloc[wire_num] = f'wire {wire_num}'
while sig <= window_max - 1:
    if overall_list[sig][4] != f'wire {wire_num}':
        wire_num += 1
        valid_df['wire num'].iloc[wire_num] = f'wire {wire_num}'
    for i in order:
        if overall_list[sig][3] == i:
            valid_df[f'{i} start'].iloc[wire_num] = overall_list[sig][1]
            valid_df[f'{i} end'].iloc[wire_num] = overall_list[sig][2]
    sig += 1

valid_df.to_csv('start_results.csv', index=False, header=True)

# #=====================For validation: calculate DTW from ASE manual=============
# ASE_manual=pd.read_csv('ASE_manual_reindex.csv', header=None, skiprows=1, usecols=list(range(6)))
# # ASE_chpoint={'1st bond':ASE_manual.iloc[1:],'looping':[],'reset motion':[]}

# =====================Save classified pictures in clus_directory==================
# encoder_channel.append('piezo')
for ch in encoder_channel:
    # create folder for store data
    clus_directory = 'clutering_' + str(ch)
    try:
        os.makedirs(clus_directory, exist_ok=True)
        print("Directory '%s' created successfully" % clus_directory)
        files = glob.glob(clus_directory + '/' + '*.png')
        for file in files:
            os.remove(file)
    except OSError as error:
        print("Directory '%s' can not be created" % clus_directory)
#
#     plot_change_points(signal=df, ts_change_loc=new_loc_id, \
#                        save_fig=True, filename='0_overall.png', channel=[ch], directory=clus_directory)
    plot_change_points(signal=df, ts_change_loc=new_loc_id, \
                       save_fig=True, filename='all_channel.png', channel=encoder_channel, directory=clus_directory)
#     if ch == 'z':
#         plot_max_min(piezo=df[ch_col['piezo']], encoder=df[ch_col[ch]], ts_change_loc=change_loc_sum_id, \
#                      low_envelope_indices_id=low_envelope_indices_id, high_envelope_indices_id=high_envelope_indices_id, \
#                      file_name='reference_for_engineering.png', directory=clus_directory)
#
#     for sig in overall_freq:
#         plt.figure(figsize=(16, 5), dpi=150)
#         plt.plot(df[ch_col[ch]].loc[overall_freq[sig][0]:overall_freq[sig][1]])
#         y_locator = MultipleLocator(0.5)
#         ax = plt.gca()
#         ax.yaxis.set_major_locator(y_locator)
#         plt.ylim(-1, 4)
#         plt.savefig(clus_directory + '/' + str(sig) + '_' + str(overall_freq[sig][2]))



end = time.time()
print(f'running time: {end - start} sec')
print('Thank you for using the prgram!')
