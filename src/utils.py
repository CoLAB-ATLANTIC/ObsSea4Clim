import os
import pickle
import shutil
import cv2

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
        "ignore", 
        category=RuntimeWarning, 
        message="invalid value encountered in cast"
    )
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="invalid value encountered in divide"
)

from absl import logging
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap, BoundaryNorm

import src.config as config
import dev.projections as prj

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
    

def get_latlon_range_as_string(dataset):
    """
    Retrieves the latitude and longitude range from an xarray dataset and formats it as a string.

    Parameters:
        dataset (xarray.Dataset): The dataset containing latitude and longitude dimensions.

    Returns:
        str: A formatted string of the latitude and longitude range.
    """
    # Retrieve latitude and longitude ranges
    lat_min = dataset.lat.min().item(); lat_max = dataset.lat.max().item()
    lon_min = dataset.lon.min().item(); lon_max = dataset.lon.max().item()

    # Format the string with cardinal directions
    lat_range = f"{abs(lat_min):.0f}°{'S' if lat_min < 0 else 'N'} to {abs(lat_max):.0f}°{'S' if lat_max < 0 else 'N'}"
    lon_range = f"{abs(lon_min):.0f}°{'W' if lon_min < 0 else 'E'} to {abs(lon_max):.0f}°{'W' if lon_max < 0 else 'E'}"

    return f"{lat_range}, {lon_range}"


def get_input_data(data_path, start_date, end_date, latlon):

    start_year = int(datetime.strptime(start_date, '%Y-%m-%d').year)
    end_year = int(datetime.strptime(end_date, '%Y-%m-%d').year)

    # THIS CAN BE AUTOMATIC, WHERE WE CHECK IF THE PATH IS A FOLDER OR A .NC FILE
    # NO NEED FOR  SINGLE_INPUT FLAG
    if config.FLAGS.single_input:
        files = [data_path]
    else:
        files = [data_path + 'mhw_hobday_data' + f'/mhw_{year}.nc' for year in range(start_year, end_year+1)]
    data = xr.open_mfdataset(files)#, chunks={'time': 200}

    varname = config.FLAGS.varname
    data=data[[varname]]

    # select time and latlon interval
    data = data.sel(time = slice(start_date, end_date))

    if 'latitude' in data.coords: data = data.rename({'latitude': 'lat'})
    if 'longitude' in data.coords: data = data.rename({'longitude': 'lon'})

    if latlon:
        lat_min, lat_max, lon_min, lon_max = parse_latlon_range(latlon)
        data = data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    ratio = config.DOWNSAMPLE_RATIO
    if ratio != 1 and ratio > 0:
        old_shape = data[varname].shape[1:]
        data = data.isel(lat=slice(0, None, ratio), lon=slice(0, None, ratio))
        print(f'Downsampled gridded data {ratio} times: from (n_lat,n_lon) = {old_shape} to {data[varname].shape[1:]}')

    dummy_ds = xr.Dataset(coords = data.coords).drop_vars("time")
    dummy_ds.to_netcdf(config.INTERNAL_DATA_PATH+'dummy_latlon.nc')
    del dummy_ds

    logging.info(f"Lat/Lon range: {get_latlon_range_as_string(data)}")
    logging.info(f"Projecting from Degrees to Sinusoidal...")
    kmres = config.KM_RESOLUTION*ratio
    data, x, y = prj.reproject_raster(data, varname, cell_size_km=kmres)

    return data, x, y


def quantize_data(data, min_value):
    data = xr.where((data < min_value) | np.isnan(data), 0, 1)
    return data.astype(np.uint8)


def manage_last_label(reset):
    file_path = config.LABEL_FILE

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
    possible_lbls = list(range(last_lbl+1)) 

    # Check if the mapping file exists
    if not os.path.exists(config.MAPPING_FILE):
        if last_lbl == 0: label_mapping[1]=1
        else:
            # Initialize the mapping file
            label_mapping = {last_lbl: last_lbl}

            for lbl in possible_lbls:
                label_mapping[lbl]=lbl

    else:
        # Load the existing mapping file
        with open(config.MAPPING_FILE, 'rb') as f:
            label_mapping = pickle.load(f)
        
        lbl_keys = list(label_mapping.keys())
        
        for lbl in possible_lbls:
            if lbl not in lbl_keys:
                label_mapping[lbl]=lbl

    with open(config.MAPPING_FILE, 'wb') as f:
        pickle.dump(label_mapping, f)


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
    

def compute_num_windows(full_window: int, small_window: int, step_size: int) -> int:
    """
    Compute the number of windows given the full window length, 
    small window size, and step size.
    
    Parameters:
    - full_window: Total size of the data
    - small_window: Size of each individual window
    - step_size: Shift between consecutive windows
    
    Returns:
    - Number of windows as an integer
    """
    if small_window > full_window:
        raise ValueError("Small window size must be less than or equal to the full window size.")
    if step_size <= 0:
        raise ValueError("Step size must be a positive integer.")
    
    return (full_window - small_window) // step_size + 1


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

    #save reprojected for later use
    dummy_ds = xr.open_dataset(config.INTERNAL_DATA_PATH+'dummy_latlon.nc')
    save_final_chunk = prj.reproject_raster_back2latlon(new_chunk.copy(), dummy_ds, 'label')
    save_final_chunk['label'] = save_final_chunk['label'].astype(np.uint8)
    save_final_chunk.to_netcdf(config.INTERNAL_DATA_PATH+f'Events_{name}', compute=True)
    save_final_chunk.close()

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


def save_output_nc(data_dict, outpath, x, y): 
    varname = config.FLAGS.varname
    dummy_ds = xr.open_dataset(config.INTERNAL_DATA_PATH+'dummy_latlon.nc')

    # Create a new xarray dataset
    data_array = xr.DataArray(data_dict['Areas'].astype(np.uint8),
                              dims=['time', 'y', 'x'],
                              coords={'time': data_dict['time_array'],
                                       'x': x, 'y': y})
    
    ds = xr.Dataset({varname: data_array}); del data_array

    ds = prj.reproject_raster_back2latlon(ds, dummy_ds, varname)
    ds[varname] = ds[varname].astype(np.uint8)

    ds = xr.Dataset({'event': ds[varname],
                          'pixel_sum': data_dict['pixel_sum'],
                          'ID': data_dict['ID'],
                          'pixel_area_km': config.KM_RESOLUTION**2
                          })
    
    # Save to a new NetCDF file
    outfolder = f'{outpath}events/'
    os.makedirs(outfolder, exist_ok=True)
    output_file = outfolder + data_dict['ID'] + '.nc'
    ds.to_netcdf(output_file)
    ds.close()


def get_serial_number(number, maximum = 10_000):
    num_digits = len(str(maximum)) + 1
    serial_number = str(number).zfill(num_digits)
    return serial_number

def get_month(month):
    month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    return month_name[month-1]

def compute_total_area(num_pixels, side_resolution_km):
    """
    Compute the total occupied area in square kilometers given the number of pixels 
    and the side resolution of each pixel.
    
    Parameters:
        num_pixels (int): The total number of pixels.
        side_resolution_km (float): The side length of each pixel in kilometers.
    
    Returns:
        float: Total occupied area in square kilometers.
    """
    total_area = num_pixels * (side_resolution_km ** 2)
    return total_area