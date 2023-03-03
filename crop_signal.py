from fileinput import filename
import pandas as pd
import glob
import os
import argparse
import numpy as np
from statistics import median 
import warnings
warnings.filterwarnings("ignore")
import ruptures as rpt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import signal
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
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
        if i==0:
            crop_window.append([0, change_location[i]])
            continue

        crop_window.append([change_location[i-1], change_location[i]])
        
    return crop_window

def classify_signal(seq_encoder, change_loc, low_envelope_indices,high_envelope_indices, encoder_threshold):
      
    signal_state = []
    change_loc = [loc for loc in change_loc if loc[0]!=loc[1]]

    for idx, val in enumerate(change_loc):
        encoder_signal = seq_encoder[val[0]: val[1]]
        low_idx_interval = low_envelope_indices[(low_envelope_indices >= val[0]) & (low_envelope_indices <= val[1])]
        high_idx_interval = high_envelope_indices[(high_envelope_indices >= val[0]) & (high_envelope_indices <= val[1])]

        if ((np.mean(seq_encoder[high_idx_interval])-np.mean(seq_encoder[low_idx_interval]))>encoder_threshold):
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
        if idx==(len(signal_state)-1):
            new_loc.append(change_loc[idx][1])
            signal_state_removed.append(val) 
       
        current_state = val
    return new_loc, make_crop_index(new_loc), signal_state_removed
def remove_close_crop(change_loc, signal_state):
    new_loc = [0]
    signal_state_removed=[signal_state[0]]
    close_threshold=int(params['threshold_outlier'])/20
    for idx, val in enumerate(change_loc):
        if idx == 0: continue
        if (change_loc[idx]-new_loc[-1])>close_threshold:
            new_loc.append(change_loc[idx])
            signal_state_removed.append(signal_state[idx])
    return new_loc, make_crop_index(new_loc), signal_state_removed
def extend_envelope(df_np, env_indices_np):  
    extend_list = []

    for idx, val in enumerate(env_indices_np):
        if idx == len(env_indices_np)-1:
            extend_list.extend([val for _ in range(len(df_np) - val)])
            return np.array(extend_list)
        if idx == 0 and env_indices_np[0] != 0:
            extend_list.extend([val for _ in range(val)])
        else:
            extend_list.extend([val for _ in range(env_indices_np[idx+1] - val)])

def plot_change_points(piezo, encoder, ts_change_loc, save_fig, filename, directory,color='red', alpha=1):

    # encoder.index = range(skiprows_num,skiprows_num+nrows_num)
    # ts_change_loc=np.array([x+skiprows_num for x in ts_change_loc])
    # change_loc_piezo=np.array([x+skiprows_num for x in change_loc_piezo])
    # change_loc_z=np.array([x+skiprows_num for x in change_loc_z])

    plt.figure(figsize=(100, 5), dpi=150)
    plt.plot(encoder,label='encoder')
    plt.plot(piezo,label='piezo')
    for x in ts_change_loc:
        #plt.axvline(x, lw=2, linestyle='--', color=color, alpha=alpha)
        if (x==change_loc_piezo_id).any():
            plt.axvline(x, lw=2, linestyle='--', color='red', alpha=1)
            continue
        elif (x==change_loc_z_id).any():
            plt.axvline(x, lw=2, linestyle='--', color='black', alpha=1)
    plt.plot([], [], label='change location piezo', linestyle='--', color='red')
    plt.plot([], [], label='change location z encoder', linestyle='--', color='black')
    plt.legend(loc="upper right")     
    if save_fig:
        plt.savefig(directory +'/'+filename)
    plt.ylim(-1,4)
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
    
    # plt.plot(x_sampled)
    # plt.plot(x.values)

    for i in range(len(preference_2)):
        pref_normalized = normalize_signal(preference_2[i])
        pref_sampled = sampling_data(pref_normalized)
        dist_low, path2 = fastdtw(x_sampled, pref_sampled, dist=euclidean)
        cumulative_score += dist_low

    return cumulative_score

#=====================Plot high and low frequency pictures==============
def plot_high_low(encoder_channel):
    for ch in encoder_channel:
        directory = 'cropped_signal_'+str(ch)
        try:
            os.makedirs(directory, exist_ok = True)
            print("Directory '%s' created successfully" % directory)
            files = glob.glob(directory+'/'+'*.png')
            for file in files:
                os.remove(file)
        except OSError as error:
            print("Directory '%s' can not be created" % directory)
        plot_change_points(piezo=df[piezo_col], encoder=df[globals()[str(ch)+'_col']], ts_change_loc=new_loc_id,\
             save_fig=True, filename='overall.png',directory=directory)
        plot_change_points(piezo=df[piezo_col], encoder=df[globals()[str(ch)+'_col']], ts_change_loc=change_loc_sum_id, \
        save_fig=True, filename='origianl change points.png',directory=directory)
        for idx, val in enumerate(new_crop_id):
            plt.figure(figsize=(16, 5), dpi=150) 
            plt.ylim(-1,4)                
            # if df[globals()[str(ch)+'_col']].loc[val[0]: val[1]].values.shape == (0,):
            if val[0]==val[1]:
                continue
            plt.plot(df[globals()[str(ch)+'_col']].loc[val[0]: val[1]])
            if len(df[globals()[str(ch)+'_col']].loc[val[0]: val[1]].values) > int(params['threshold_outlier']):
                plt.savefig(directory + '/outlier_' + str(idx))           
            else:
                plt.savefig(directory + '/' +signal_state_removed[idx] + '_' + str(idx))
            plt.close()


# ====================================================================

parser = argparse.ArgumentParser(
    prog='crop_signal.py',
    description="Python program to automatic dividing encoder/piezo signals."
    )

parser.add_argument('input_data', 
                    type=str, 
                    help='path to ".csv" file')

# parser.add_argument('chunk_size',
#                     nargs='?',
#                     default=1000000,
#                     type=int,
#                     help='define chunk_size to load data chunk by chunk.')

# parser.add_argument('threshold_piezo',
#                     nargs='?',
#                     type=float,
#                     default=1.85,
#                     help='mean value of piezo sensor signal interval at a high level (bonding moving stage)')
parser.add_argument('skiprows_num', 
                    nargs='?', 
                    default=1,
                    type=int,
                    help='skiprows')
parser.add_argument('nrows_num', 
                    nargs='?',
                    type=int,
                    help='nrows')
input_path = parser.parse_args().input_data
skiprows_num = parser.parse_args().skiprows_num
nrows_num=parser.parse_args().nrows_num

# chunk_size = parser.parse_args().chunk_size
# threshold_piezo = parser.parse_args().threshold_piezo

# =======================================================================
# =======================================================================

# create config file 
if not os.path.exists('configure.txt'):
    details = {'chunk_size': 1000000,
               'rolling_size': 200,
               'sampling_rate': 80,
               'change_sensitive': 2.5,
               'threshold_piezo': 1.85,
               'threshold_outlier': 10000,
               'encoder_variance': 2,
               'max_min_encoder': 1.5}


    with open("configure.txt", 'w') as f:
        for key, value in details.items():
            f.write('%s:%s\n' % (key, value))

params = {}
with open('configure.txt', 'r') as config:
    for line in config:
        params[line.partition(':')[0]] = line.partition(':')[2][:-1].strip()

# =======================================================================
change_location = []
encoder_channel=[]
second=[]
second_pref=[]
high_freq= {}
low_freq={}
dtw_score={}
# ====================Read file for crop====================================
df_summary = pd.read_csv(input_path, header=None, iterator=True, chunksize=int(params['chunk_size']),\
    skiprows=skiprows_num,nrows=nrows_num)
df=pd.concat([chunk for chunk in df_summary],ignore_index=True).reset_index().drop('index',axis=1)
print(f'data length: {len(df)}')
print(df.head())
print(df.tail())
# ====================Read default golden sample====================================
golden1 = pd.read_csv('golden_1.csv', header=None).iloc[:,3]
golden2 = pd.read_csv('golden_2.csv', header=None).iloc[:,3]
# ====================Assign channel column=============================================
print('Please count the columns starting from 0')
piezo_col=int(input(f'The column of Piezo: ')) 
for ch in ['x','y','z']:
    globals()[str(ch)+'_col']=int(input(f'The column of {ch}_encoder (-1 if no need to crop) :')) #Define x_col,y_col,z_col
    if globals()[str(ch)+'_col'] != -1:
        encoder_channel.append(ch)  
# print(f'{*encoder_channel,} channel will be cropped.')
# ====================================================================
seq_piezo=df[piezo_col]
for ch in encoder_channel:
    globals()[str(ch)+'_seq_encoder'] = df[globals()[str(ch)+'_col']]
# =======================================================================
indices_z = [i for i in range(0, 0+len(seq_piezo), int(params['sampling_rate_z']))]
indices_piezo = [i for i in range(0, 0+len(seq_piezo), int(params['sampling_rate_piezo']))]

# Detect the change points for encoder data
z_normalize = butter_highpass_filter(data=globals()['z_seq_encoder'], cut=0.005, order=2)

# print(time.time()-start,'butter')
sampling_z = np.take(z_normalize, indices_z)

# print(time.time()-start,'sampling z')
alg_z = rpt.Pelt(model="rbf").fit(sampling_z)

# print(time.time()-start,'algz')
change_loc_z = np.array(alg_z.predict(pen=float(params['change_sensitive_z'])))*int(params['sampling_rate_z'])
# print(time.time()-start,'loc_z')

# detect change point for piezo sensor data
sampling_piezo = pd.DataFrame(seq_piezo).rolling(window=int(params['rolling_size']), min_periods=1).mean() 
sampling_piezo = np.take(sampling_piezo.values, indices_piezo)
alg_piezo = rpt.Pelt(model="rbf").fit(sampling_piezo)
change_loc_piezo = np.array(alg_piezo.predict(pen=float(params['change_sensitive_piezo'])))*int(params['sampling_rate_piezo'])

# print(time.time()-start,'loc_piezo')
change_loc_sum = np.unique(np.sort(np.concatenate((change_loc_z, change_loc_piezo), axis=None)))
crop_window = make_crop_index(change_loc_sum)


low_envelope_indices, high_envelope_indices = hl_envelopes_idx(globals()['z_seq_encoder'].values\
    , dmin=int(params['dmin']), dmax=int(params['dmax']))
print('dmax',int(params['dmax']))
print('dmin',int(params['dmin']))
# print(params)
# print(int(params['sampling_rate_piezo']))
# print(time.time()-start,'lmax')
extend_low_envelope = extend_envelope(globals()['z_seq_encoder'].values, low_envelope_indices)
# print(time.time()-start,'extend')
extend_high_envelope = extend_envelope(globals()['z_seq_encoder'].values, high_envelope_indices)
signal_state = classify_signal(globals()['z_seq_encoder'].values, crop_window, extend_low_envelope,extend_high_envelope,encoder_threshold=2.0)
new_loc_a, new_crop_a, signal_state_removed_a = remove_redundant_loc(crop_window, signal_state) 
new_loc_b, new_crop_b, signal_state_removed_b = remove_close_crop(new_loc_a, signal_state_removed_a) 
new_loc, new_crop, signal_state_removed = remove_redundant_loc(new_crop_b, signal_state_removed_b) 

#=====================Reindex of data and change points==================
df.index = range(skiprows_num,skiprows_num+len(seq_piezo))
new_crop_id=np.array(new_crop)
new_crop_id=new_crop_id+skiprows_num
new_loc_id=np.array([x+skiprows_num for x in new_loc])
change_loc_piezo_id=np.array([x+skiprows_num for x in change_loc_piezo])
change_loc_z_id=np.array([x+skiprows_num for x in change_loc_z])
change_loc_sum_id=np.array([x+skiprows_num for x in change_loc_sum])
# #=====================Save high and low frequency pictures==============
# for ch in encoder_channel:
#     directory = 'cropped_signal_'+str(ch)
#     try:
#         os.makedirs(directory, exist_ok = True)
#         print("Directory '%s' created successfully" % directory)
#         files = glob.glob(directory+'/'+'*.png')
#         for file in files:
#             os.remove(file)
#     except OSError as error:
#         print("Directory '%s' can not be created" % directory)
#     plot_change_points(piezo=df[piezo_col], encoder=df[globals()[str(ch)+'_col']], ts_change_loc=new_loc_id, save_fig=True, directory=directory)
#     while save_high_low_pic:
#         for idx, val in enumerate(new_crop_id):
#             plt.figure(figsize=(16, 5), dpi=150)                 
#             # if df[globals()[str(ch)+'_col']].loc[val[0]: val[1]].values.shape == (0,):
#             if val[0]==val[1]:
#                 continue
#             plt.plot(df[globals()[str(ch)+'_col']].loc[val[0]: val[1]])
#             if len(df[globals()[str(ch)+'_col']].loc[val[0]: val[1]].values) > int(params['threshold_outlier']):
#                 plt.savefig(directory + '/outlier_' + str(idx))           
#             else:
#                 plt.savefig(directory + '/' +signal_state_removed[idx] + '_' + str(idx))
#             plt.close()
#         print(f'High and low amplitude in {ch} saved')
#         break
#=====================Determine high and low frequency==================
for idx, val in enumerate(new_crop_id):
    if val[0]==val[1]:
        continue
    if signal_state_removed[idx]=='high' and len(df[piezo_col].loc[val[0]: val[1]].values) <= int(params['threshold_outlier']):
        high_freq[idx] = [val[0],val[1]]
    elif signal_state_removed[idx]=='low' and len(df[piezo_col].loc[val[0]: val[1]].values) <= int(params['threshold_outlier']):
        low_freq[idx] = [val[0],val[1]]

#=================Define class method===================
class_method=str(params['class_method'])
#=================DTW===================================
if class_method=='DTW':
    #=====================Clssify 1st bond/looping/2nd bond==================        
    ans = input(f'Do you want to select golden sample automatically (Type N for manual mode):')
    no = ['No', 'no', 'n', 'N']
    yes = ['Yes', 'yes', 'Y', 'y']
    condition = True
    while condition:
        if ans in no:
            print('Manual mode')
            plot_high_low(['z'])
            print('Please select golden sample from reset motion in cropped_siganl_z folder.')
            # enter the index of looping and reset motion
            num_pref = int(input(f'How many golden sample do you want to compare:'))
            for i in range(num_pref):
                second.append(int(input(f'Index of sample {i+1} for reset motion: ')))
            for key in second:
                second_pref.append(df[globals()[('z_col')]].loc[high_freq[key][0]:high_freq[key][1]])
            break
        elif ans in yes:
            second_pref.append(golden1)
            second_pref.append(golden2)
            print('Auto mode')
            break
        else:
            ans = input('Please enter Yes or No: ')
    for sig in high_freq:           
        dtw_score[sig]=compare_signal(df[globals()[('z_col')]].loc[high_freq[sig][0]:high_freq[sig][1]],second_pref, sig-1)
        # print('dtw_score',dtw_score)
        # print('dtw_score.values',dtw_score.values)
        # print(np.array(list(dtw_score.values())))
    threshold_dtw=np.median(np.array(list(dtw_score.values())))
    print('threshold_dtw',threshold_dtw)

    print('Comparing cropped signals with the reference signal ... ')
    for sig in high_freq:       
        print(f'The distance of {sig} example with pref is: {dtw_score[sig]}')
        if dtw_score[sig] >= threshold_dtw:
            high_freq[sig].append('looping')
            if sig-1 in low_freq:
                low_freq[sig-1].append('1st bond')
            if (sig+2 not in high_freq):
                if sig+1 in low_freq:
                    low_freq[sig+1].append('2nd bond')
        elif dtw_score[sig] < threshold_dtw:
            high_freq[sig].append('reset motion')
            if sig-1 in low_freq:
                low_freq[sig-1].append('2nd bond')
            if (sig+2 not in high_freq):
                if sig+1 in low_freq:
                    low_freq[sig+1].append('1st bond')
    for sig in low_freq:
        print('sig',sig,low_freq[sig])
        if len(low_freq[sig])==2:
            low_freq[sig].append('low freq not classified')
#=================Piezo===================
elif class_method=='piezo':
    for sig in low_freq:
        # print('min_piezo',np.min(df[piezo_col].loc[low_freq[sig][0]:low_freq[sig][1]]))
        if np.mean(df[piezo_col].loc[low_freq[sig][0]:low_freq[sig][1]])<float(params['threshold_piezo']):
            low_freq[sig].append('1st bond')
            if sig-1 in high_freq:
                high_freq[sig-1].append('reset motion')
            if sig+2 not in low_freq:
                if sig+1 in high_freq:
                    high_freq[sig+1].append('looping')
        else:  
            low_freq[sig].append('2st bond')
            if sig-1 in high_freq:
                high_freq[sig-1].append('looping')
            if sig+2 not in low_freq:
                if sig+1 in high_freq:
                    high_freq[sig+1].append('reset motion')
    for sig in high_freq:
        # print('sig','high',sig,high_freq[sig])
        if len(high_freq[sig])==2:
            high_freq[sig].append('high freq not classified')

overall_freq={**high_freq,**low_freq}
# print('overall',overall_freq)
#=====================Save classified results in raw_directory==================        
# create folder
raw_directory = 'clutering_raw_data' 
try:
    os.makedirs(raw_directory, exist_ok = True)
    print("Directory '%s' created successfully" % raw_directory)
    files = glob.glob(raw_directory+'/'+'*')
    for file in files:
        os.remove(file)
except OSError as error:
    print("Directory '%s' can not be created" % raw_directory)
for sig in overall_freq:
    # print(f'The {sig} crop is identified as {overall_freq[sig][2]}')
    df.loc[overall_freq[sig][0]:overall_freq[sig][1],:].\
        to_csv(raw_directory+'/'+str(overall_freq[sig][2])+'_'+str(sig)+'.csv',index=False,header=False) 
#=====================Save classified pictures in clus_directory================== 
encoder_channel.append('piezo')
for ch in encoder_channel:
# create folder for store data
    clus_directory = 'clutering_'+str(ch)
    try:
        os.makedirs(clus_directory, exist_ok = True)
        print("Directory '%s' created successfully" % clus_directory)
        files = glob.glob(clus_directory+'/'+'*.png')
        for file in files:
            os.remove(file)
    except OSError as error:
        print("Directory '%s' can not be created" % clus_directory)
    plot_change_points(piezo=df[piezo_col], encoder=df[globals()[str(ch)+'_col']], ts_change_loc=new_loc_id, \
        save_fig=True, filename='overall.png',directory=clus_directory)
    # plot_change_points(piezo=df[piezo_col], encoder=df[globals()[str(ch)+'_col']], ts_change_loc=change_loc_sum_id, \
    #     save_fig=True, filename='origianl change points.png',directory=clus_directory)
    for sig in overall_freq:              
        plt.figure(figsize=(16, 5), dpi=150)
        plt.plot(df[globals()[(str(ch)+'_col')]].loc[overall_freq[sig][0]:overall_freq[sig][1]])
        plt.ylim(-1,4)
        plt.savefig(clus_directory + '/' + str(overall_freq[sig][2]) + '_' + str(sig))


end = time.time()    
print(f'running time: {end-start} sec')
print('Thank you for using the prgram!')       

