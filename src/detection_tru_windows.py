import os
import numpy as np
from datetime import datetime, timedelta

import src.config as config
import src.utils as utils
import src.detection_utils as det

from src.config import FLAGS

DEBUG_MEMORY = False

def detection_tru_windows(ds, last_lbl):

    start_date = datetime.strptime(FLAGS.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(FLAGS.end_date, '%Y-%m-%d')

    window_size = config.WINDOW_SIZE
    step_size = config.STEP_SIZE

    window_start_day = start_date
    window_end_day = window_start_day + window_size

    output_folder = config.OUTPUT_PATH
    internal_folder = config.INTERNAL_DATA_PATH
    video_folder = config.VIDEO_FOLDER
    os.makedirs(video_folder, exist_ok=True)

    varname = FLAGS.varname

    """ if params.getp('reset'):
        if os.path.exists(mapping_dir): os.remove(mapping_dir)
        outpath = output_folder + 'mhw_dataset.nc'
        if os.path.exists(outpath): remove_all_files_in_folder(output_folder) """

    w=1
    n_days = (end_date - start_date).days + 1
    n_windows = utils.compute_num_windows(n_days,
                                           window_size.days,
                                             step_size.days)

    while window_start_day < end_date:
        current_time = datetime.now().strftime('%H:%M')
        print(f'window ({w}/{n_windows+1}): {window_start_day} to {window_end_day}; last label: {last_lbl} [{current_time}]')
        w+=1

        if DEBUG_MEMORY: utils.memory_print('cycle beginning')

        #detect mhws in current window
        current_window_nc = ds.sel(time = slice(window_start_day, window_end_day))
        current_window = current_window_nc[varname].values
        
        if DEBUG_MEMORY: utils.memory_print('before detect_mhws')
        current_window, last_lbl = det.detect_mhws_2d(current_window, last_lbl, verbose = config.VERBOSE)
        current_window_nc['label'] = (('time', 'y', 'x'), current_window)
        current_window_nc = current_window_nc.drop_vars(varname)
        del current_window
        if DEBUG_MEMORY: utils.memory_print('after detect_mhws')
        
        if window_start_day != start_date:   
            #get overlap windows
            current_overlap = current_window_nc.sel(time = slice(overlap_start, overlap_end))
            current_overlap = current_overlap['label'].values
            previous_overlap = previous_overlap['label'].values
            
            if DEBUG_MEMORY: utils.memory_print('before update_overlap')
            overlap_window, label_mapping = det.update_overlap(previous_overlap, current_overlap)
            del previous_overlap
            if DEBUG_MEMORY: utils.memory_print('after update_overlap')
            
            current_window = current_window_nc.sel(time = slice(overlap_end + timedelta(days=1),
                                                                 window_end_day))
            current_window = current_window['label'].values
            current_window = det.relabel_data_window(current_window, label_mapping)
            if DEBUG_MEMORY: utils.memory_print('after relabel_data_window')
            
            current_window = np.concatenate((overlap_window, current_window))
            current_window_nc['label'] =  (('time', 'y', 'x'), current_window)
            del current_window
            if DEBUG_MEMORY: utils.memory_print('after concat window')
        
        window_start_day_prev = window_start_day
        window_end_day_prev = window_end_day
        
        #exception for the last timestep
        if window_end_day >= end_date:
            lst_time = current_window_nc.time.values
            print(f'LAST window: {utils.to_datetime(lst_time[0])} to {utils.to_datetime(lst_time[-1])};\
                   last label: {last_lbl}')
            str1=utils.to_datetime(lst_time[0]).strftime('%Y-%m-%d')
            str2=utils.to_datetime(lst_time[-1]).strftime('%Y-%m-%d'); del lst_time
            name = f'{str1}_to_{str2}'

            if config.RENDER_VIDEOS:
                print('Saving video')
                utils.save_video(current_window_nc.label.values, video_folder + name + '.mp4', fps=5)

            utils.save_window(current_window_nc, output_folder, name + '.nc', separate=True)
            with open(config.LABEL_FILE, 'w') as f: f.write(str(last_lbl))

            del current_window_nc
            break
        else:
            window_start_day += step_size
            window_end_day += step_size

            if DEBUG_MEMORY: utils.memory_print('before saving window')
            
            previous_window_save = current_window_nc.sel(time = slice(window_start_day_prev,
                                                                window_start_day-timedelta(days=1)))
            
            str1=window_start_day_prev.strftime('%Y-%m-%d')
            str2=window_start_day-timedelta(days=1); str2=str2.strftime('%Y-%m-%d')
            name = f'{str1}_to_{str2}'
            
            if config.RENDER_VIDEOS:
                print('Saving video')
                utils.save_video(previous_window_save.label.values, video_folder + name + '.mp4', fps=5)
            
            utils.save_window(previous_window_save, output_folder, name + '.nc',  separate=True)
            del previous_window_save
            if DEBUG_MEMORY: utils.memory_print('after saving window')

            with open(config.LABEL_FILE, 'w') as f: f.write(str(last_lbl))
        
        # Find the overlapping date range
        overlap_start, overlap_end = utils.find_overlap_dates(window_start_day_prev, window_end_day_prev,
                                                        window_start_day, window_end_day)
        
        previous_overlap = current_window_nc.sel(time = slice(overlap_start, overlap_end))
        utils.backup_overlap(internal_folder, 'previous_overlap.nc', 'previous_overlap_backup.nc')
        utils.save_window(previous_overlap, internal_folder, 'previous_overlap.nc',  separate=True)
        with open(config.LABEL_FILE, 'w') as f: f.write(str(last_lbl))
        del current_window_nc
    