import xarray as xr
import numpy as np
import os
import re
from datetime import datetime

import pandas as pd
import pickle

#to get an entire dictionary (it will occupy alot of memory)
def get_files_with_substring(folder_path, substring):
    # List to store the file names
    file_list = []

    # Loop through each file in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file name contains the substring
        if substring in file_name:
            file_list.append(file_name)
    
    return file_list

# Function to extract the serial number from the filename
def extract_serial_number(filename):
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return -1  # In case no serial number is found

# Function to extract the year from the filename
def extract_year(filename):
    match = re.search(r'_(\d{4})_[\w-]+_', filename)
    if match:
        return int(match.group(1))
    return -1  # In case no year is found

# Function to filter files based on the year range
def filter_files_by_year(file_list, start_year, end_year):
    filtered_files = [filename for filename in file_list if start_year <= extract_year(filename) <= end_year]
    return filtered_files

#convert date to datetime
def to_datetime(date):
    if type(date)==type(np.datetime64('1970-01-01T00:00:00')):
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                    / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)
    else: return date

#convert mhw in nc file to dict with Areas, Start, End, pixel_sum, ID
def from_nc2dict(nc_file, convert2string = False):
    dataset = xr.open_dataset(nc_file)

    array3d = dataset['mhw_zone'].values
    time = dataset.time.values

    # Convert to a list of sets with coordinate tuples
    coordinates_list = []
    for frame in array3d:
        coords = np.argwhere(frame == 1)
        coords_set = {tuple(coord) for coord in coords}
        coordinates_list.append(coords_set)

    del array3d

    mhw=dict()
    if convert2string: mhw['Areas'] = str(coordinates_list)
    else: mhw['Areas'] = coordinates_list
    
    mhw['pixel_sum'] = int(dataset['pixel_sum'])
    mhw['ID'] = str(np.array(dataset['ID']))
    mhw['Start'] = to_datetime(time[0]).strftime('%Y-%m-%d')
    mhw['End'] = to_datetime(time[-1]).strftime('%Y-%m-%d')
    
    del time
    dataset.close()
    
    return mhw