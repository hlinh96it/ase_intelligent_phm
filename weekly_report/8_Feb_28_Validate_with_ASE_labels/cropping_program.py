import numpy as np
from scipy import signal
from peakdetect import peakdetect
from scipy.signal import savgol_filter
import ruptures as rpt

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 3.5)
plt.rcParams['figure.dpi'] = 150

import warnings
warnings.filterwarnings('ignore')


def butter_highpass(cut=0.5, order=5):
    b, a = signal.butter(order, cut, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cut, order=5):
    b, a = butter_highpass(cut, order=order)
    y = signal.filtfilt(b, a, data, axis=0)

    return y

def peak_detection(array, look_ahead=500, diff_threshold=0.01):
    # lookahead is the distance to look ahead from a peak to determine if it is the actual peak.
    peaks = peakdetect(array, lookahead=look_ahead, delta=diff_threshold)
    higherPeaks = np.array([ii[0] for ii in peaks[0]], dtype=int)
    lowerPeaks = np.array([ii[0] for ii in peaks[1]], dtype=int)
    
    # modified for the case piezo drop not so significantly
    higherPeaks_new = []
    for ii, high_peak in enumerate(higherPeaks):
        current_look_ahead = array[high_peak: high_peak + look_ahead]
        look_ahead_dif = np.absolute(current_look_ahead - array[high_peak])
        no_significant_index = look_ahead_dif[look_ahead_dif < diff_threshold].shape[0]
        higherPeaks_new.append(no_significant_index+high_peak)

    peak_lists_ = np.concatenate([higherPeaks_new, lowerPeaks])

    return np.sort(peak_lists_)

def change_slope_detection(array, window=500, poly=3, deriv=3):
    der2 = savgol_filter(array, window_length=window, polyorder=poly, deriv=deriv)
    max_der2 = np.max(np.abs(der2))
    large = np.where(np.abs(der2) > max_der2 / 2)[0]
    gaps = np.diff(large) > window
    begins = np.insert(large[1:][gaps], 0, large[0])
    ends = np.append(large[:-1][gaps], large[-1])
    changes = ((begins + ends) / 2).astype(np.int32)

    return changes

def crop_connection(piezo):
    dary = piezo.rolling(window=200, min_periods=1).mean().values
    dary -= np.average(dary)

    # using cumulative sum method
    dary_step = np.cumsum(dary)
    dary_step = (dary_step - dary_step.min()) / (dary_step.max() - dary_step.min())
    crop_idx = peak_detection(dary_step, look_ahead=100, diff_threshold=0.01)
    # crop_idx = change_slope_detection(dary_step, 500, 2, 2)

    # insert 0 and end points
    crop_idx = np.insert(crop_idx, [0, crop_idx.shape[0]], [0, crop_idx.shape[0]])


    # TODO: Create crop pair instead of just cutting indices
    crop_idx_pair = []
    for idx, val in enumerate(crop_idx):
        try:
            crop_idx_pair.append([val, crop_idx[idx + 1]])
        except IndexError:
            pass

    crop_idx_pair = np.array(crop_idx_pair, dtype=int)
    
    return crop_idx_pair

def crop_looping_part(array, piezo, cropped_index, cropped_location, group_count, previous_state,
                      result_folder, current_idx, require_visualize=False):
    
    current_state = None
    index_1, index_2, index_3 = None, None, None

    z_normalize = abs(butter_highpass_filter(data=array, cut=0.1, order=10))

    dary_step_normalized = np.cumsum(z_normalize)
    dary_step_normalized = (dary_step_normalized - dary_step_normalized.min()) / \
                            (dary_step_normalized.max() - dary_step_normalized.min())
                            
    # ignore 1st bond parts which in low variance and frequency
    if np.std(array) < 0.5:
        current_state = '1st_bond'
        previous_state = current_state

        # save start and end location for 1st bond
        cropped_location['1st_bond_start'] = array.index[0]
        cropped_location['1st_bond_end'] = array.index[-1]
        
        if require_visualize:
            plt.plot(range(array.index[0], array.index[-1]+1), array)
            plt.plot(range(array.index[0], array.index[-1]+1), piezo)
            plt.savefig(result_folder + '/final_result_' + str(current_idx) + '.png')
            plt.close()
        
    else:
        current_state = 'looping_reset_motion'

        # TODO: how to choose change_detection method?
        change_for_an_interval = change_slope_detection(dary_step_normalized, window=50, poly=2, deriv=2)
        change_for_an_interval = np.insert(change_for_an_interval, 0, 0)

        # TODO: ignore noisy intervals
        # or just treat noisy intervals as looping
        length = change_for_an_interval - np.roll(change_for_an_interval, 1)
        top_2_length = length.argsort()[-2:][::-1]
        looping_index = max(top_2_length)

        # TODO: cropping for looping part
        index_1, index_3 = change_for_an_interval[looping_index - 1], change_for_an_interval[looping_index]
        second_bond = array[index_1: index_3].values
        index_1, index_3 = index_1 + array.index[0], index_3 + array.index[0]

        alg_z = rpt.Pelt(model="rbf").fit(second_bond)
        change_loc_z = np.array(alg_z.predict(pen=50))[0]
        index_2 = change_loc_z + index_1

        # visualize results
        if require_visualize:
            plt.plot(range(array.index[0], array.index[-1]+1), array)
            plt.plot(range(array.index[0], array.index[-1]+1), piezo)
            # plt.plot(range(index_1, index_3), second_bond, c='green')
            plt.axvline(index_1, linewidth=3.0, c='black')  # for cropping looping part
            plt.axvline(index_2, linewidth=3.0, c='r')  # for cv2
            plt.axvline(index_3, linewidth=3.0, c='black')  # for reset motion part
            plt.savefig(result_folder + '/final_result_' + str(current_idx) + '.png')
            plt.close()
        
    if current_state == previous_state:
        cropped_location['1st_bond_start'] = np.nan
        cropped_location['1st_bond_end'] = np.nan

    cropped_location['looping_start'] = array.index[0]
    cropped_location['looping_end'] = index_1
    cropped_location['cv2_start'] = index_1
    cropped_location['cv2_end'] = index_2
    cropped_location['2nd_bond_start'] = index_2
    cropped_location['2nd_bond_end'] = index_3
    cropped_location['reset_motion_start'] = index_3
    cropped_location['reset_motion_end'] = array.index[-1]

    cropped_index[group_count] = cropped_location
    cropped_location = {}
    group_count += 1
    
    return cropped_index, cropped_location, group_count

