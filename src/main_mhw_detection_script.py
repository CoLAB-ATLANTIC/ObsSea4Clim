import imports_and_functions as imf

downsample_ratio = 1

params = imf.InputParameters(
    window_size = imf.timedelta(days=728), #approx 2 years
    step_size = imf.timedelta(days=546), #approx 18 months (three quarters of the window_size)
    first_year=1982, last_year=2022,
    min_intensity=0.2, min_days=5,
    min_pixels = int(250_000/(downsample_ratio**2)),
    min_frame = int(10_000/(downsample_ratio**2)),
    output_folder = '/media/eoserver/AnaO_ATL/JP_ATL/mhw_labeled_dataset/2d_detection/',
    reset=False,
    neighbours= [[1,1,1],
                 [1,1,1],
                 [1,1,1]],
    prov_list = ['NADR', 'NASE', 'NASW', 'NATR',
             'CARB', 'NECS', 'SARC', 'ARCT',
             'GFST', 'NWCS', 'BPLR', 'CNRY'],
    MASK_PROVINCES=False,
    longhurst_path = '/home/eoserver/beatriz/JP/PROVINCES/NA_longhurst.shp'
    )            


files = [f'/media/eoserver/AnaO_ATL/JP_ATL/mhw_data/mhw_{year}.nc' for year in range(params.getp('first_year'), params.getp('last_year')+1)]
data = imf.xr.open_mfdataset(files)#, chunks={'time': 200}

#Not included in the longhurst file:
# [CAMR, GUIA, WTRA, ETRA, GUIN, MEDI]

if params.getp('MASK_PROVINCES'):
    merged_gdf = imf.get_gdf_merged_provinces(params)
else: merged_gdf=None

data = imf.preprocess_nc(data, params.getp('min_intensity'), merged_gdf)


time = imf.pd.to_datetime(data['time'].values).to_pydatetime()
first_day = time[0]
last_day = time[-1]
del time

window_start_day = first_day
window_end_day = first_day + params.getp('window_size')

step_size = params.getp('step_size')
last_lbl=1

mapping_dir = '/home/eoserver/beatriz/JP/data/label_mapping_total.pkl'

#choose the detection method
detect_mhws = [imf.detect_mhws_2d, imf.detect_mhws_cc3d][0]


#downsample nc file just for developing
data = imf.downsample_netcdf(data, ratio=downsample_ratio)
print(f'(lat, lon) size: ({len(data.lat.values)}, {len(data.lon.values)})')


################################################################################################################
 #if you want to start mid process (already have some years processed and you want to pick up the process)
params.setp('reset', False)
folder_path = params.getp('output_folder')
previous_overlap = imf.xr.open_dataset(folder_path + 'previous_overlap.nc')

previous_overlap_time = imf.pd.to_datetime(previous_overlap['time'].values).to_pydatetime()
overlap_start = previous_overlap_time[0]
overlap_end = previous_overlap_time[-1]
window_start_day = overlap_start
window_end_day = window_start_day + params.getp('window_size')

last_lbl = previous_overlap['mhw_label'].max().item() +1
#################################################################################################################

debug_memory=False

if params.getp('reset'):
    if imf.os.path.exists(mapping_dir): imf.os.remove(mapping_dir)
    outpath = params.getp('output_folder') + 'mhw_dataset.nc'
    if imf.os.path.exists(outpath): imf.remove_all_files_in_folder(params.getp('output_folder'))

while window_start_day < last_day:
    current_time = imf.datetime.now().strftime('%H:%M')
    print(f'window: {window_start_day} to {window_end_day}; last label: {last_lbl} [{current_time}]') 
    if debug_memory: imf.memory_print('cycle beginning')

    #detect mhws in current window
    current_window_nc = data.sel(time = slice(window_start_day, window_end_day))
    current_window = current_window_nc.intensity.values
    
    if debug_memory: imf.memory_print('before detect_mhws')
    current_window, last_lbl = detect_mhws(current_window, last_lbl , params)
    current_window_nc['mhw_label'] = (('time', 'lat', 'lon'), current_window)
    current_window_nc = current_window_nc.drop_vars('intensity')
    del current_window
    if debug_memory: imf.memory_print('after detect_mhws')
    
    if window_start_day != first_day:   
        #get overlap windows
        current_overlap = current_window_nc.sel(time = slice(overlap_start, overlap_end))
        current_overlap = current_overlap['mhw_label'].values
        previous_overlap = previous_overlap['mhw_label'].values
        
        if debug_memory: imf.memory_print('before update_overlap')
        overlap_window, label_mapping = imf.update_overlap(previous_overlap, current_overlap)
        del previous_overlap
        if debug_memory: imf.memory_print('after update_overlap')
        
        current_window = current_window_nc.sel(time = slice(overlap_end + imf.timedelta(days=1), window_end_day))
        current_window = current_window['mhw_label'].values
        current_window = imf.relabel_data_window(current_window, label_mapping)
        if debug_memory: imf.memory_print('after relabel_data_window')
        
        current_window = imf.np.concatenate((overlap_window, current_window))
        current_window_nc['mhw_label'] =  (('time', 'lat', 'lon'), current_window)
        del current_window
        if debug_memory: imf.memory_print('after concat window')
    
    window_start_day_prev = window_start_day
    window_end_day_prev = window_end_day
    
    if window_end_day >= last_day:
        lst_time = current_window_nc.time.values
        print(f'LAST window: {imf.to_datetime(lst_time[0])} to {imf.to_datetime(lst_time[-1])}; last label: {last_lbl}')
        str1=imf.to_datetime(lst_time[0]).strftime('%Y-%m-%d')
        str2=imf.to_datetime(lst_time[-1]).strftime('%Y-%m-%d'); del lst_time
        name = f'{str1}_to_{str2}'
        imf.save_window(current_window_nc, params.getp('output_folder'), name + '.nc',  separate=True) #exception for the last timestep
        del current_window_nc
        break
    else:
        if debug_memory: imf.memory_print('before saving window')
        window_start_day += step_size
        window_end_day += step_size
        
        previous_window_save = current_window_nc.sel(time = slice(window_start_day_prev,
                                                               window_start_day-imf.timedelta(days=1)))
        
        str1=window_start_day_prev.strftime('%Y-%m-%d')
        str2=window_start_day-imf.timedelta(days=1); str2=str2.strftime('%Y-%m-%d')
        name = f'{str1}_to_{str2}'
        
        video_folder = '/home/eoserver/beatriz/JP/mhw_detection_jun_version/videos/no_prov/'
        imf.save_video(previous_window_save.mhw_label.values, video_folder + name + '.mp4', fps=5)
        
        imf.save_window(previous_window_save, params.getp('output_folder'), name + '.nc',  separate=True)
        del previous_window_save
        if debug_memory: imf.memory_print('after saving window')
    
    # Find the overlapping date range
    overlap_start, overlap_end = imf.find_overlap_dates(window_start_day_prev, window_end_day_prev,
                                                    window_start_day, window_end_day)
    
    previous_overlap = current_window_nc.sel(time = slice(overlap_start, overlap_end))
    imf.backup_overlap(params.getp('output_folder'),'previous_overlap.nc','previous_overlap_backup.nc')
    imf.save_window(previous_overlap, params.getp('output_folder'), 'previous_overlap.nc',  separate=True)
    del current_window_nc

#add label mapping for labels that didnt appear in the overlap
imf.adjust_label_mapping(last_lbl)