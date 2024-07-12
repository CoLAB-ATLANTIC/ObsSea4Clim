import geopandas as gpd
import numpy as np
import pickle
from shapely.geometry import Point
import os
import xarray as xr

class MHW_Class:
    def __init__(self, mhws, starts, ends, open_lbls, del_lbls):
        self.mhws = mhws
        self.starts = starts
        self.ends = ends
        self.open_lbls = open_lbls
        self.del_lbls = del_lbls
        
def get_dates_and_cut(remaining_labels, labels_out, Start, params, chunk_mhws):
    for segid in remaining_labels:
        extracted_image = labels_out * (labels_out == segid)
        
        first_timestep, last_timestep = get_timeframe(extracted_image, segid)
        Start_mhw = to_datetime(Start) + timedelta(days = first_timestep)
        End_mhw = to_datetime(Start) + timedelta(days = last_timestep)
        
        if last_timestep-first_timestep >= params.getp('min_duration'):
            chunk_mhws.mhws[segid] = extracted_image[first_timestep:last_timestep+1]
            chunk_mhws.starts[segid] = Start_mhw; chunk_mhws.ends[segid] = End_mhw
    return chunk_mhws

def save_or_close_mhws(chunk_mhws, open_mhws, start_date, end_date):
    for mhw_label, mhw_array in chunk_mhws.mhws.items():
        if mhw_label in open_mhws.open_lbls:
            if chunk_mhws.starts[mhw_label] != to_datetime(start_date):
                print('discontinuity error: mhw with specific label has at least one day in between without signal')
            
            if chunk_mhws.ends[mhw_label] < to_datetime(end_date):
                open_mhws.open_lbls.remove(mhw_label)
                chunk_mhws.mhws[mhw_label] = np.concatenate((open_mhws.mhws[mhw_label], chunk_mhws.mhws[mhw_label]))
                chunk_mhws.starts[mhw_label] = open_mhws.starts[mhw_label]
                
                if len(chunk_mhws.mhws[mhw_label])-1 != (chunk_mhws.ends[mhw_label] - chunk_mhws.starts[mhw_label]).days:
                    print(f'start and end date dont match size of 3darray: array is {len(chunk_mhws.mhws[mhw_label])-1} and dates are {(chunk_mhws.ends[mhw_label] - chunk_mhws.starts[mhw_label]).days}')
                
                #save_video(chunk_mhws.mhws[mhw_label], folder_videos +fileid1 + '_in_prev' + '.mp4')
                del open_mhws.mhws[mhw_label]; del open_mhws.starts[mhw_label]; del open_mhws.ends[mhw_label]
            else:
                chunk_mhws.mhws[mhw_label] = np.concatenate((open_mhws.mhws[mhw_label], chunk_mhws.mhws[mhw_label]))
                open_mhws.ends[mhw_label] = chunk_mhws.ends[mhw_label]
                        
        if chunk_mhws.ends[mhw_label] == to_datetime(end_date):
            if mhw_label not in open_mhws.open_lbls:
                open_mhws.open_lbls.append(mhw_label)
                open_mhws.starts[mhw_label] = chunk_mhws.starts[mhw_label]
                open_mhws.mhws[mhw_label] = mhw_array
            else:
                open_mhws.mhws[mhw_label] = np.concatenate((open_mhws.mhws[mhw_label], mhw_array))
            
            open_mhws.ends[mhw_label] = chunk_mhws.ends[mhw_label]
            chunk_mhws.del_lbls.append(mhw_label)
            
    return chunk_mhws, open_mhws     


def save_last_open(chunk_mhws, open_mhws):
    with open( './open_labels.pkl', 'wb') as outf:
        pickle.dump(list(open_mhws.mhws.keys()), outf)
    for key, value in open_mhws.mhws.items():
        if key not in list(chunk_mhws.mhws.keys()):
            chunk_mhws.mhws[key] = value
            chunk_mhws.starts[key] = open_mhws.starts[key]
            chunk_mhws.ends[key] = open_mhws.ends[key]
        else:
            chunk_mhws.mhws[key] = np.concatenate((value, chunk_mhws.mhws[key]))
            chunk_mhws.starts[key]=open_mhws.starts[key]
    return chunk_mhws


def convert_coordinates(x, y , step_size= 0.05, coord_limits=[-89.975, 9.975, 10.025, 59.975]):
    min_lon, max_lon, min_lat, max_lat = coord_limits
    #y, x = pixel_coords
    lon = min_lon + y * step_size
    lat = min_lat + x * step_size
    
    lon = round(lon, 4); lat = round(lat, 4)
    return lat, lon

def get_province_asarray(map_bounds=(1000, 2000)):
    prov_array_dir = '/home/eoserver/beatriz/JP/data/'+'prov_arrays.pkl'
    if map_bounds == (1000, 2000) and os.path.exists(prov_array_dir):
        with open(prov_array_dir, 'rb') as f:
            prov_arrays = pickle.load(f)
    else:
        prov_arrays = compute_province_asarray(prov_array_dir, map_bounds)
    return prov_arrays

def compute_province_asarray(prov_array_dir, map_bounds = (1000, 2000)):
    longhurst_path = '/home/eoserver/beatriz/JP/PROVINCES/NA_longhurst.shp'
    longhurst = gpd.read_file(longhurst_path)
    
    prov = ['NADR', 'NATR', 'NASW', 'NASE']
    prov_arrays = {}

    for province in prov:
        # Create an empty 2D array with zeros
        map_array = np.zeros((map_bounds[0], map_bounds[1]))
        
        province_gdf = longhurst[longhurst['ProvCode']==province].iloc[0]

        # Iterate over each pixel in the 2D map
        for x in range(map_bounds[0]):
            for y in range(map_bounds[1]):
                # Convert pixel coordinates to coordinates
                lat, lon = convert_coordinates(x, y)
                
                # Check if the coordinates fall within the province polygon
                if province_gdf['geometry'].contains(Point(lon, lat)):
                    # Set the pixel value to 1
                    map_array[x, y] = 1
                    
        prov_arrays[province] = map_array
    
    if map_bounds == (1000, 2000):
        with open(prov_array_dir, 'wb') as outfile:
            pickle.dump(prov_arrays, outfile)
            
    return prov_arrays


def get_serial_number(number, maximum=10_000):
    num_digits = len(str(maximum))+1
    serial_number = str(number).zfill(num_digits)
    return serial_number

def get_month(month):
    month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    return month_name[month-1]

def most_occur_NAO(start_date, end_date, df):
    # Initialize counters for positive and negative values
    signal_count = 0
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create a date range from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        year = current_date.astype('datetime64[Y]').astype(int) + 1970
        month = current_date.astype('datetime64[M]').astype(int) % 12 + 1
        month_name = months[month - 1]
        
        # Check if the year and month exist in the DataFrame
        if year in df['Year'].values:
            row = df[df['Year'] == year]
            if month_name in row.columns:
                value = row[month_name].values[0] #; print(f'{current_date} : {value}')
                if value > 0: signal_count += 1
                elif value < 0: signal_count -= 1
        
        # Move to the next month
        if month == 12: current_date = np.datetime64(f'{year + 1}-01-01')
        else: current_date = np.datetime64(f'{year}-{month + 1:02d}-01')

    if signal_count > 0: most_occurring_signal = 'POS'
    elif signal_count < 0: most_occurring_signal = 'NEG'
    else: most_occurring_signal = 'OSC'
        
    return most_occurring_signal

def get_province_label(frames, prov_arrays):
    #GET PROVINCE LABELS
    label_count = 0
    label_prov = 'NA'
    for province, prov_mask in prov_arrays.items():
        mhws_province = frames*prov_mask
        unique, counts = np.unique(mhws_province, return_counts=True)
        
        count=0
        for uniq_lbl, c in zip(unique, counts):
            if uniq_lbl > 0: count+=c 

        if label_count < count:
            label_prov=province
            label_count=count
    return label_prov

def get_mhw_ids(frames, start_date, end_date, prov_arrays, NAO_df, serial_number, total_pixels):
    province = get_province_label(frames, prov_arrays)

    serial_number = get_serial_number(serial_number)
    year = start_date.year
    month = start_date.month
    
    NAO = most_occur_NAO(np.datetime64(start_date), np.datetime64(end_date), NAO_df)
    ID = province + '_' + NAO + '_' + str(year) + '_' + get_month(month) + '_' + serial_number
    
    new_dict = dict()
    new_dict['ID'] = ID
    new_dict['Start'] = start_date
    new_dict['End'] = end_date
    
    new_dict['Areas'] = frames.astype(np.uint8); del frames
    new_dict['pixel_sum'] = total_pixels
    
    return new_dict

def save_output_nc(data_dict, params, latitudes, longitudes):
    output_file = params.getp('mhw_output_folder') + data_dict['ID'] + '.nc'
    output_file_backup = '/media/eoserver/AnaO_ATL/JP_ATL/MHW_DATASET_JUN_2D_DETECTION/' + data_dict['ID'] + '.nc'
    
    
    # Create a new xarray dataset
    data_array = xr.DataArray(data_dict['Areas'].astype(np.uint8),
                              dims=['time', 'lat', 'lon'],
                              coords={'time': data_dict['time_array'], 'lat': latitudes, 'lon': longitudes})
    
    ds = xr.Dataset({'mhw_zone': data_array,
                          'pixel_sum': data_dict['pixel_sum'],
                          'ID': data_dict['ID']})
    
    # Save to a new NetCDF file
    ds.to_netcdf(output_file)
    ds.to_netcdf(output_file_backup)
    ds.close()