import pickle

import numpy as np
import xarray as xr

from datetime import timedelta

import src.utils as utils
import src.detection_utils as detect
import src.config as config

# REMOVE THIS IF POSSIBLE! CHECK!!!
check_overlap=False

def get_timesteps_per_area(folder_path):
    ''' go through the mhw windows and get the start and end date of each mhw,
    so that we can cut them (separate) afterwards
    '''

    file_identifier = '_to_'
    files_with_substring = utils.get_files_with_substring(folder_path, file_identifier)
    files = [folder_path + file for file in files_with_substring]

    dataset = xr.open_mfdataset(files, chunks={'time': 200})
    time_ds = dataset.time.values

    cut_dates = dict() #record start, end, and cut stamps of mhws

    prev_frame = dataset.sel(time=time_ds[0])['label'].values
    prev_unique = list(np.unique(prev_frame))
    if check_overlap: prev_unique.remove(0)

    for lbl in prev_unique:
        if lbl not in cut_dates.keys():
            cut_dates[lbl] = list()
            cut_dates[lbl].append(utils.to_datetime(time_ds[0]))

    for date in time_ds[1:]:
        date = utils.to_datetime(date)
        #print(date.strftime('%Y-%m-%d'))

        frame = dataset.sel(time=date)['label'].values

        frame_lbls = list(np.unique(frame))

        if check_overlap:
            aux_labels=frame_lbls.copy(); aux_labels.remove(0)
            #exception for last date: close every date
            if date==utils.to_datetime(time_ds[-1]):   
                for aux_lbl in aux_labels:
                    if aux_lbl in cut_dates.keys():
                        cut_dates[aux_lbl].append(date)
            
            # get new start_dates
            for lbl in aux_labels:
                if lbl not in cut_dates.keys():
                    cut_dates[lbl] = list()
                    cut_dates[lbl].append(date)

            #encode combination of all possible labels
            unique_codes, _ = detect.get_unique_code(prev_unique, frame_lbls)
            #get the pairing count for each label and total_overlap
            label_match, total_overlap, _, _, _ = detect.get_pairing_count(prev_frame, frame, prev_unique, frame_lbls, unique_codes)

            # get new end dates #
            for lbl, overlap_count in label_match.items():
                # overlap condition OR mhw ended in prev_frame
                if total_overlap[lbl]==0 or overlap_count/total_overlap[lbl] < 0.5:   #or overlap_count< min_pixels_per_frame
                    cut_dates[lbl].append(date-timedelta(days=1))
        else:
            frame_lbls.remove(0)
            
            #open labels
            for lbl in frame_lbls:
                if lbl not in cut_dates.keys():
                    cut_dates[lbl] = list()
                    cut_dates[lbl].append(date)
            
            #close labels   
            for lbl in prev_unique:
                if lbl not in frame_lbls:
                    cut_dates[lbl].append(date-timedelta(days=1))
                    
            if date == time_ds[-1]:
                for lbl in frame_lbls:
                    cut_dates[lbl].append(date)
                

        ###################################################################################### 
        prev_frame = frame
        prev_unique = frame_lbls
            
    dataset.close()

    if 0 in cut_dates.keys(): del cut_dates[0] #just in case

    if check_overlap: lbldates_filename = 'lbl_dates.pkl'
    else: lbldates_filename = 'lbl_dates_no_overlap.pkl'
    lbl_dates_dir = config.INTERNAL_DATA_PATH + lbldates_filename
 
    with open(lbl_dates_dir, 'wb') as outf:
        pickle.dump(cut_dates, outf)


def get_event_id(frames, start_date, end_date, serial_number, total_pixels):
    serial_number = utils.get_serial_number(serial_number)
    year = start_date.year
    month = start_date.month
    
    ID = 'EV_' + serial_number + '_' + utils.get_month(month) + '_' + str(year)
    
    new_dict = dict()
    new_dict['ID'] = ID
    new_dict['Start'] = start_date
    new_dict['End'] = end_date
    
    new_dict['Areas'] = frames.astype(np.uint8); del frames
    new_dict['pixel_sum'] = total_pixels
    
    return new_dict


def splice_and_id_events(x, y):
    lbl_dates_dir = f'{config.INTERNAL_DATA_PATH}lbl_dates_no_overlap.pkl'
    with open(lbl_dates_dir, 'rb') as f: lbl_dates = pickle.load(f)
    labels = lbl_dates.keys()

    folder_path = config.OUTPUT_PATH  # replace with your folder path
    substring = '_to_'
    files_with_substring = utils.get_files_with_substring(folder_path, substring)
    files = [folder_path + file for file in files_with_substring]

    dataset = xr.open_mfdataset(files, chunks={'time': 200})

    min_pixels_time = detect.calculate_num_pixels(config.FLAGS.min_area_time, config.KM_RESOLUTION)
    min_pixels_time = int(min_pixels_time/(config.DOWNSAMPLE_RATIO**2))

    serial_number = 1
    n_labels = len(labels)
    for i, lbl in enumerate(list(labels)):
        start_date = utils.to_datetime(lbl_dates[lbl][0])
        for end_date in lbl_dates[lbl][1:]:
            end_date = utils.to_datetime(end_date)
            if (end_date-start_date).days >= config.FLAGS.min_days:
                str1 = start_date.strftime('%Y-%m-%d'); str2 = end_date.strftime('%Y-%m-%d')
                
                chunk = dataset.sel(time=slice(start_date, end_date))
                time_range = chunk.time.values
                
                frames = chunk['label'].values; del chunk
                frames = np.where(frames!=lbl, 0, 1)
                
                total_pixels = np.count_nonzero(frames)
                if total_pixels >= min_pixels_time:
                    area_km = utils.compute_total_area(total_pixels, config.KM_RESOLUTION)
                    print(f'Event {serial_number} from label {lbl} ({i+1}/{n_labels}): from {str1} to {str2}; area: {area_km:,} km²')
                    
                    event_data = get_event_id(frames, start_date, end_date, serial_number, total_pixels)
                    serial_number += 1
                    
                    event_data['time_array'] = time_range
                    utils.save_output_nc(event_data, folder_path, x, y)
                        
                    del event_data
                
                del frames
            start_date = end_date + timedelta(days=1)

    dataset.close()



