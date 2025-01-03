import os
import pickle
import shutil
import cv2

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta

import src.config as config

def reset_folders():
    if os.path.exists(config.OUTPUT_PATH):
        shutil.rmtree(config.OUTPUT_PATH)
    if os.path.exists(config.INTERNAL_DATA_PATH):
        shutil.rmtree(config.INTERNAL_DATA_PATH)

    os.makedirs(config.OUTPUT_PATH)
    os.makedirs(config.INTERNAL_DATA_PATH)

    last_label_path = config.INTERNAL_DATA_PATH + 'last_label.txt'
    if not os.path.exists(last_label_path):
        with open(last_label_path, "w") as file:
            file.write("0")


def get_files_with_substring(folder_path, substring):
    # List to store the file names
    file_list = []

    # Loop through each file in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file name contains the substring
        if substring in file_name:
            file_list.append(file_name)
    
    return file_list


# Function to validate and convert the lat/lon range to a tuple of floats
def parse_latlon_range(values):
    if len(values) != 4:
        raise ValueError("You must provide exactly 4 values: lat_min lat_max lon_min lon_max.")
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, values)
        if lat_min > lat_max or lon_min > lon_max:
            raise ValueError("Min lat/lon values cannot be greater than max values.")
        return lat_min, lat_max, lon_min, lon_max
    except ValueError as e:
        raise ValueError(f"Invalid lat/lon range: {e}")


def get_input_data(data_folder, start_date, end_date, latlon):

    start_year = int(datetime.strptime(start_date, '%Y-%m-%d').year)
    end_year = int(datetime.strptime(end_date, '%Y-%m-%d').year)

    files = [data_folder + 'mhw_hobday_data' + f'/mhw_{year}.nc' for year in range(start_year, end_year+1)]

    data = xr.open_mfdataset(files)#, chunks={'time': 200}

    # select time and latlon interval
    data = data.sel(time = slice(start_date, end_date))
    if latlon:
        lat_min, lat_max, lon_min, lon_max = parse_latlon_range(latlon)
        data = data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    ratio = config.DOWNSAMPLE_RATIO
    if ratio > 1:
        old_shape = data.intensity.shape[1:]
        data = data.isel(lat=slice(0, None, ratio), lon=slice(0, None, ratio))
        print(f'Downsampled gridded data {ratio} times: from (lat,lon) = {old_shape} to {data.intensity.shape[1:]}')

    return data


def quantize_data(data, min_intensity):

    data = data.drop_vars('category')
    data = xr.where((data < min_intensity) | np.isnan(data), 0, 1)

    return data.astype(np.uint8)


def manage_last_label(reset, file_path='./data/internal/last_label.txt'):
    # Check if the file exists, if not, create it with a default value of 1
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('1')

    if reset:
        # If reset is True, set last_label to 1 and update the file
        last_label = 1
        with open(file_path, 'w') as file:
            file.write(str(last_label))
    else:
        # If reset is False, read the last_label from the file
        with open(file_path, 'r') as file:
            last_label = int(file.read().strip())

    return last_label


def adjust_label_mapping(last_lbl):

    with open(config.MAPPING_FILE, 'rb') as f:
        label_mapping = pickle.load(f)

    possible_lbls = list(range(last_lbl+1)) 
    lbl_keys = list(label_mapping.keys())
    
    for lbl in possible_lbls:
        if lbl not in lbl_keys:
            label_mapping[lbl]=lbl

    with open( config.MAPPING_FILE, 'wb') as outf:
        pickle.dump(label_mapping, outf)


def to_datetime(date):
    if type(date)==type(np.datetime64('1970-01-01T00:00:00')):
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                    / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)
    else: return date


# Function to find overlapping date range
def find_overlap_dates(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start < overlap_end:
        return overlap_start, overlap_end
    else:
        return None, None


def check_consistency(existing_dataset, new_chunk):
    last_day_existing = to_datetime(existing_dataset.time.values[-1])
    first_day_chunk = to_datetime(new_chunk.time.values[0])
    
    if last_day_existing+timedelta(days=1) != first_day_chunk:
        print(f'ERROR: dates are not consistent. Tried to connect {last_day_existing} with {first_day_chunk}')
        os.exit()


def save_window(new_chunk, folder, name='filename', separate = False):
    if separate:
        filepath = folder + name
    else:
        filepath= folder + 'mhw_dataset.nc'
        if os.path.exists(filepath):
            existing_dataset = xr.open_dataset(filepath, chunks={'time': 200})
            
            check_consistency(existing_dataset, new_chunk)
            
            # Concatenate along the time dimension
            new_chunk = xr.concat([existing_dataset, new_chunk], dim='time')
            existing_dataset.close()

    if os.path.exists(filepath): os.remove(filepath)
    new_chunk.to_netcdf(filepath, compute=True)
    new_chunk.close()


def backup_overlap(src_path, old_name, new_name):
    # Construct the full paths for the source and destination files
    src_file_path = os.path.join(src_path, old_name)
    dest_file_path = os.path.join(src_path, new_name)

    # Ensure the source file exists before attempting to copy
    if not os.path.exists(src_file_path):
        print(f"The source file {src_file_path} does not exist.")
        return

    # Copy the file to the new destination with the new name
    shutil.copy(src_file_path, dest_file_path)


def runtime_steps(start_times, end_times, prepend_str):
    # Calculate total time
    steps = list(start_times.keys())
    total_time = sum(end_times[step] - start_times[step] for step in steps)

    # Create the result string
    result_str = prepend_str
    for step in steps:
        step_time = end_times[step] - start_times[step]
        percentage = (step_time / total_time) * 100
        result_str += f"{step} {step_time:.2f}s ({percentage:.2f}%), "

    # Remove the trailing comma and space
    result_str = result_str.rstrip(", ")

    # Print the result string
    print(result_str)


def upsample_array(array, ratio):
    """
    Upsample a 2D array by a given ratio.
    
    Parameters:
        array (np.ndarray): The input 2D array.
        ratio (int): The upsampling factor.
        
    Returns:
        np.ndarray: The upsampled 2D array.
    """
    if ratio < 1:
        raise ValueError("Ratio must be 1 or greater.")
    
    # Repeat rows and columns by the ratio
    upsampled_array = np.repeat(np.repeat(array, ratio, axis=0), ratio, axis=1)
    return upsampled_array


def draw_dynamic_text(frame_shape, ref_size=(1000, 2000), ref_pos=(50, 70), ref_font_scale=2, ref_thickness=6):
    """
    Draws text on a frame with dynamically adjusted position, font size, and thickness 
    based on the frame size relative to a reference size.
    
    Parameters:
        frame (numpy.ndarray): The image/frame to draw the text on.
        text (str): The text to draw.
        ref_size (tuple): Reference frame size as (height, width).
        ref_pos (tuple): Reference position (x, y) for the text.
        ref_font_scale (float): Reference font scale.
        ref_thickness (int): Reference thickness.
    
    Returns:
        numpy.ndarray: The frame with the text drawn on it.
    """
    # Current frame size
    current_height, current_width = frame_shape
    
    # Calculate scaling factors for width and height
    height_ratio = current_height / ref_size[0]
    width_ratio = current_width / ref_size[1]
    
    # Use the smaller ratio to maintain proportionality
    scale_factor = min(height_ratio, width_ratio)
    
    # Adjust position
    x = int(ref_pos[0] * width_ratio)
    y = int(ref_pos[1] * height_ratio)
    position = (x, y)
    
    # Adjust font scale and thickness
    font_scale = ref_font_scale * scale_factor
    thickness = max(1, int(ref_thickness * scale_factor))  # Ensure thickness is at least 1
    
    return position, font_scale, thickness


# Function to save video from 2D frames
def save_video(frames, output_file, fps=5):
    height, width = upsample_array(frames[0].copy(), config.DOWNSAMPLE_RATIO).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        labels = list(np.unique(frame))
        if 0 in labels: labels.remove(0)

        frame=upsample_array(frame, config.DOWNSAMPLE_RATIO)
        frame_aux=frame
        
        # Ensure the frame is uint8 and in the range [0, 255]
        frame = np.float64(frame); frame /= np.max(frame)
        frame *= 255; frame = np.uint8(frame)

        # Convert to 3-channel
        frame_3d = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        position, font_scale, thickness = draw_dynamic_text(frame.shape[:2])
        
        cv2.putText(frame_3d, str(i), position,
                     cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 255, 0), thickness)

        # Loop over each label and draw the label at the centroid
        for label in labels:  # Start from 1 to skip the background
            indices = np.argwhere(frame_aux == label)
            centroid = indices.mean(axis=0)
            x_centroid, y_centroid = int(centroid[1]), int(centroid[0])
            label_text = str(label)

            # Draw the label text at the centroid
            cv2.putText(frame_3d, label_text, (x_centroid, y_centroid), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        out.write(frame_3d)

    out.release()
    cv2.destroyAllWindows()


def get_colors(last_lbl):
    #plot colors just for debugging
    n_labels = last_lbl+40
    colors = plt.cm.viridis(np.linspace(0, 1, n_labels))  # You can choose any colormap you like
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_labels+0.5, 1), ncolors=n_labels)
    return cmap, norm


def plot_frame(frame, txt, cmap, norm, with_text=True):
    if with_text:
        labels = list(np.unique(frame))
        if 0 in labels: labels.remove(0)

        for label in labels:
            indices = np.argwhere(frame == label)
            centroid = indices.mean(axis=0)
            x_centroid, y_centroid = int(centroid[1]), int(centroid[0])
            label_text = str(label)

            plt.text(x_centroid, y_centroid, label_text, color='red', ha='center', va='center')
        
    plt.imshow(frame, cmap=cmap, norm=norm); plt.title(txt); plt.show()


def save_output_nc(data_dict, outpath, latitudes, longitudes):

    outfolder = f'{outpath}events/'
    os.makedirs(outfolder, exist_ok=True)

    output_file = outfolder + data_dict['ID'] + '.nc'
    
    # Create a new xarray dataset
    data_array = xr.DataArray(data_dict['Areas'].astype(np.uint8),
                              dims=['time', 'lat', 'lon'],
                              coords={'time': data_dict['time_array'],
                                       'lat': latitudes, 'lon': longitudes})
    
    ds = xr.Dataset({'mhw_zone': data_array,
                          'pixel_sum': data_dict['pixel_sum'],
                          'ID': data_dict['ID']})
    
    # Save to a new NetCDF file
    ds.to_netcdf(output_file)
    ds.close()


def get_serial_number(number, maximum=10_000):
    num_digits = len(str(maximum))+1
    serial_number = str(number).zfill(num_digits)
    return serial_number

def get_month(month):
    month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    return month_name[month-1]