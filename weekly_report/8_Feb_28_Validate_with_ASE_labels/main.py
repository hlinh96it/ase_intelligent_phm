# %% import libraries
import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

import pandas as pd
import matplotlib.pyplot as plt

from cropping_program import *

plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 3.5)
plt.rcParams['figure.dpi'] = 150

import warnings
warnings.filterwarnings('ignore')


def read_program(label_df_group, data_folder, validate_folders, program_no, drop_cols, test=False):
    program = os.listdir(os.path.join(data_folder, validate_folders[program_no], 'One_Die'))
    all_program = [csv_file for csv_file in program if csv_file.endswith('.csv')]
    cropp_index_col = label_df_group.get_group(validate_folders[program_no][2:])['rawdata']
    start, end = cropp_index_col.min().astype(int), cropp_index_col.max().astype(int)

    df = []
    if test is not False: all_program = [all_program[test]]
    for csv_file in all_program:
        csv_path = os.path.join(data_folder, validate_folders[program_no], 'One_Die', csv_file)
        df.append(pd.read_csv(csv_path, header=None, index_col=None).iloc[1:].reset_index(drop=True).drop(drop_cols, axis=1)[start: end])
    
    df = pd.concat(df, ignore_index=True)
    df.columns = ['Z', 'Piezo']
    df = df.astype(float)
    
    return df, label_df_group.get_group(validate_folders[program_no][2:]), start

def get_label(validate_file, program_no):
    validate_program = validate_file.get_group(program_no)['rawdata']
    return validate_program

def export_to_csv(group, program_name, crop_idx):
    start_label = group[group['status'] == 'Start']
    end_label = group[group['status'] == 'End']
    
    export_csv = start_label[['program', 'rawdata']].reset_index(drop=True)
    export_csv['end_label'] = end_label['rawdata'].values
    
    export_csv = pd.concat([export_csv, pd.DataFrame(crop_idx, columns=['start', 'end'])], axis=1)
    
    return export_csv


# ===============================================================================================

label_df = pd.read_excel('????????????.xlsm', sheet_name='????????????vs????????????')
label_df = label_df[(label_df['rawdata'].isna() == False) & (label_df['window_length']==101)]
label_df_group = label_df.groupby('program')  
validate_programs = [program for program, group in label_df_group]

data_folder = '/Users/hlinh96it/Library/CloudStorage/OneDrive-NTHU/ASE_PHM_WireBonding/dataset/10_DAQ_20221227'
validate_folders = os.listdir(data_folder)
validate_folders = [folder for folder in validate_folders if folder[2:] in validate_programs]

## Validate method and compare with provided labels ==================================
export_excel = None
for program_id, program_folder in tqdm(enumerate(validate_folders)):
    print(f'Processing for program {program_folder}..........')
    
    try:
        os.makedirs(os.path.join('results', program_folder))
    except FileExistsError:
        # Get a list of all files in the folder
        files = glob.glob(os.path.join(os.path.join('results', program_folder), '*'))

        # Loop through the list of files and remove each one
        for file in files:
            os.remove(file)
    
    result_folder = os.path.join('results', program_folder)
            
            
    ## Read data and Preprocessing data to better cropping ==================================
    program_data, group, start = read_program(label_df_group, data_folder, validate_folders, program_id, 
                                        drop_cols=[0, 1, 2, 5], test=0)
    piezo = program_data['Piezo'].copy()
    
    
    ## Cropping for each connection ===========================================================
    crop_idx_pair = crop_connection(piezo)
    
    if export_excel is None:
        export_excel = export_to_csv(group, program_folder[2:], crop_idx_pair[:-1, :]+start)
    else:
        export_excel = pd.concat([export_excel, export_to_csv(group, program_folder[2:], crop_idx_pair[:-1, :]+start)],
                                 axis=0)
        
    export_excel.fillna(method='ffill').to_excel('all_program_cropped.xlsx', index=False)
    
    
    ## Crop for looping parts =====================================================================
    # cropped_index, cropped_location = {}, {}
    # group_count = 1
    # previous_state = None
    
    # for idx, pair in enumerate(tqdm(crop_idx_pair)):
    #     if pair[0] - pair[1] > 0:
    #         break
        
    #     cropped_index, cropped_location, group_count = \
    #         crop_looping_part(program_data['Z'][pair[0]: pair[1]], program_data['Piezo'][pair[0]: pair[1]], 
    #                             cropped_index, cropped_location, group_count, previous_state,
    #                             result_folder, current_idx=idx)
            
    # export_cols = ['1st_bond_start', '1st_bond_end', 'looping_start', 'looping_end', 'cv2_start', 'cv2_end',
    #             '2nd_bond_start', '2nd_bond_end', 'reset_motion_start', 'reset_motion_end']
    # export_data = pd.DataFrame.from_dict(cropped_index, orient='index', columns=export_cols)
    # export_data.to_csv(result_folder + '/cropping_result.csv')

