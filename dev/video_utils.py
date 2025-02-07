import os
import pickle
import cc3d
import cv2
import re
import math
import time
import psutil
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import scipy.ndimage as scp
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from shapely.geometry import Point
from shapely.ops import unary_union
from rasterio import Affine
from rasterio.features import rasterize
from typing import Optional, Union
from pandas.plotting import table
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm

class InputParameters:
    def __init__(self, **kwargs):
        self.parameters = kwargs
        
    def setp(self, key, value):
        self.parameters[key] = value
    
    def getp(self, key):
        return self.parameters.get(key, None)
    
    def display_param(self):
        for key, value in self.parameters.items():
            print(f"{key}: {value}")
            
def memory_print(appendix):
    # Get the memory details
    memory_info = psutil.virtual_memory()
    current_time = datetime.now().strftime('%H:%M')
    print(f'{appendix}: {memory_info.used / (1024 ** 3):.2f}/{memory_info.total / (1024 ** 3):.2f} GB used [{current_time}]')
            
def to_datetime(date):
    if type(date)==type(np.datetime64('1970-01-01T00:00:00')):
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                    / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)
    else: return date

def affine_from_xr_ds(*, xr_dataset: [xr.Dataset]=None,
                      nc_path: Optional[Union[Path,str]]=None) -> Affine:
    '''Takes Xarray Dataset file and returns Affine Geotransform function
    Opionally, path to netCDF file can be provided instead of Dataset.

    NOTE:   in netCDF lon/lat values are stored in ascending order, hense the image
            construction starts from lower-left corner
    NOTE:   pixel coordinates are sampled from centroid, meaning, we should adjust
            half pixel size to both x and y axis to move the coordinates to pixel
            corner to match GDAL and Rasterio style

    Returns: rasterio Affine function
    '''
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    std_res=0.05

    if xr_dataset: ds = xr_dataset
    elif nc_path: ds = xr.open_dataset(nc_path)
    else: raise ValueError('Either Dataset or path to netCDF file need to be provided')

    xres = std_res; yres = std_res

    xy_res_in_attr = 'geospatial_lat_resolution' in ds.attrs
    if xy_res_in_attr: xres = ds.attrs['geospatial_lon_resolution']; yres = ds.attrs['geospatial_lat_resolution']

    if  not ('longitude' in ds or 'lon' in ds):
        raise AttributeError('Coordinates could not be found in attributes')

    try: # If coords are defined as longitude/latitude in the dataset
        llx = ds.longitude.values[0]; lly = ds.latitude.values[0]
        if not xy_res_in_attr: xres = ds.longitude.step; yres = ds.latitude.step
    except AttributeError: # If coords are defined as lon/lat in the dataset
        llx = ds.lon.values[0]; lly = ds.lat.values[0]
        if not xy_res_in_attr: xres = std_res; yres = std_res

    xrot = 0; yrot = 0

    return Affine(xres, xrot, llx - xres/2, yrot, yres, lly - yres/2) 


def shape2raster_mask(shapefile, xr_dataset: xr.Dataset, all_touched: bool=True) -> np.ndarray:
    
    transform = affine_from_xr_ds(xr_dataset=xr_dataset)
    width = xr_dataset.sizes['lon']; height = xr_dataset.sizes['lat']

    return rasterize(
        shapes=shapefile,
        out_shape=(height, width),
        transform=transform,
        all_touched=all_touched)

def mask_province(province, data_sel):
    mask = shape2raster_mask( province.geometry, data_sel) * 1.
    mask[mask==0] = np.nan
    data_masked = data_sel*mask
    return data_masked

def preprocess_nc(data, min_intensity, merged_gdf=None):
    data = data.drop_vars('category')
    data = xr.where((data < min_intensity) | np.isnan(data), 0, 1)
    if merged_gdf: data = mask_province(merged_gdf, data)
    data=data.astype(np.uint8)
    return data

def get_gdf_merged_provinces(params):
    longhurst_path=params.getp('longhurst_path')
    prov_list=params.getp('prov_list')
    
    longhurst = gpd.read_file(longhurst_path)

    # Filter the GeoDataFrame to select the provinces you want to merge
    provinces_to_merge = longhurst[longhurst['ProvCode'].isin(prov_list)] 

    # Merge the selected provinces into a single geometry
    merged_provinces = unary_union(provinces_to_merge.geometry)

    # Convert the merged geometry into a GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(geometry=[merged_provinces])
    return merged_gdf

def downsample_netcdf(src, ratio=2):
    # Downsample the latitude and longitude dimensions
    return src.isel(lat=slice(0, None, ratio), lon=slice(0, None, ratio))

# Function to find overlapping date range
def find_overlap_dates(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start < overlap_end:
        return overlap_start, overlap_end
    else:
        return None, None
    
def add_tuple_to_groups(tuples, groups):
    for a, b in tuples:
        # Find groups containing 'a' or 'b'
        a_group = None; b_group = None
        for group in groups:
            if a in group: a_group = group
            if b in group: b_group = group
            # If both elements are found, no need to check further
            if a_group and b_group: break
            
        if a_group and b_group: # If both elements are in different groups, merge the groups
            if a_group != b_group:
                a_group.update(b_group)
                groups.remove(b_group)
        elif a_group: # If only 'a' is in a group, add 'b' to that group
            a_group.add(b)
        elif b_group: # If only 'b' is in a group, add 'a' to that group
            b_group.add(a)
        else: # If neither element is in any group, create a new group
            groups.append({a, b})
    return groups


def update_label_mapping(label_pairing):
    mapping_dir = '/home/eoserver/beatriz/JP/data/label_mapping_total.pkl'
    if os.path.exists(mapping_dir):
        with open(mapping_dir, 'rb') as f:
            label_mapping = pickle.load(f)
    else: label_mapping = dict()
    
    if len(label_pairing)>0:
        groups = [{label_pairing[0][0]}]
        groups = add_tuple_to_groups(label_pairing, groups) #print(groups)
        
        for group_lbls in groups:
            mother_lbl = min(group_lbls)
            
            for child_lbl in group_lbls:
                if child_lbl in list(label_mapping.keys()):
                    label_mapping[child_lbl] = min(mother_lbl, label_mapping[child_lbl])
                else:
                    label_mapping[child_lbl] = mother_lbl
                    
        with open( mapping_dir, 'wb') as outf:
            pickle.dump(label_mapping, outf)
        
    return label_mapping

def decode_cantor(z):
    # Compute w
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    # Compute t
    t = w * (w + 1) // 2
    # Compute y and x
    y = z - t
    x = w - y
    return (int(x), int(y))

def create_label_pairing_new(data1, data2):
    paired_array = np.vectorize(cantor_pairing)(data1, data2)
    del data1; del data2
    
    cantor_codes = list(np.unique(paired_array)); del paired_array
    if 0 in cantor_codes: cantor_codes.remove(0)
    
    data1_unique=list(); data2_unique=list()
    label_pairing = list()
    for code in cantor_codes:
        lbl_tuple = decode_cantor(code)
        if lbl_tuple[0] != 0 and lbl_tuple[1] != 0:
            label_pairing.append(lbl_tuple)
        elif lbl_tuple[0] == 0:
            label_pairing.append((lbl_tuple[1],lbl_tuple[1]))
        else: label_pairing.append((lbl_tuple[0],lbl_tuple[0]))
        
        #save unique values
        if lbl_tuple[0] not in data1_unique and lbl_tuple[0]!=0: data1_unique.append(lbl_tuple[0])
        if lbl_tuple[1] not in data2_unique and lbl_tuple[1]!=0: data2_unique.append(lbl_tuple[1])
        
    del cantor_codes
    
    label_mapping = update_label_mapping(label_pairing); del label_pairing
    #current_time = datetime.now().strftime('%H:%M'); print(f'label mapping length: {len(label_mapping)} [{current_time}]')
    return label_mapping, data1_unique, data2_unique
    
def create_label_pairing(data1, data2):
    label_pairing = list()
    
    labels1 = list(np.unique(data1)); labels1.remove(0)
    labels2 = list(np.unique(data2)); labels2.remove(0)
    
    remaining_labels = labels1.copy() + labels2.copy()
    for label1 in labels1:
        coords1 = np.argwhere(data1 == label1)
        for coord in coords1:
            coord_idxs=tuple(coord)
            if data2[coord_idxs] != 0:
                label_pairing.append((label1, data2[coord_idxs]))
                if label1 in remaining_labels:
                    remaining_labels.remove(label1)
                    if data2[coord_idxs] in remaining_labels:
                        remaining_labels.remove(data2[coord_idxs])
                    
    for lbl in remaining_labels:
        label_pairing.append((lbl, lbl))
    
    label_mapping = update_label_mapping(label_pairing)
    return label_mapping

def adjust_label_mapping(last_lbl):
    mapping_dir = '/home/eoserver/beatriz/JP/data/label_mapping_total.pkl'
    with open(mapping_dir, 'rb') as f:
        label_mapping = pickle.load(f)

    possible_lbls = list(range(last_lbl+1)) #cuidado que não existem necessariamente todas as lbls
    lbl_keys = list(label_mapping.keys())
    
    for lbl in possible_lbls:
        if lbl not in lbl_keys:
            label_mapping[lbl]=lbl

    with open( mapping_dir, 'wb') as outf:
        pickle.dump(label_mapping, outf)
    return

def edit_labels(arr, lbl):
    arr=arr*-1 #so there is no confusion between the detected label integers and the integers we want to assign
    values = np.unique(arr)
    values = values[values != 0]
    values = sorted(values, reverse=True)
    for old_lbl in values:
        arr[arr==old_lbl]=lbl; lbl+=1
    return arr, lbl

def edit_labels_faster(arr, lbl):
    arr[arr != 0] += lbl
    lbl = arr.max()+1
    return arr, lbl

def detect_mhws_cc3d(array3d, lbl, params):
    labels = cc3d.connected_components(array3d, connectivity=6); del array3d
    labels = cc3d.dust( labels, threshold = params.getp('min_pixels'), connectivity=6, in_place=False)
    labels = labels.astype('uint16')
    labels, lbl = edit_labels(labels, lbl)
    return labels, lbl

def detect_mhws_2d(array3d, lbl, params):
    labels = detect_frame_by_frame(array3d, params, verbose= False)
    labels = labels.astype('uint16')
    labels, lbl = edit_labels(labels, lbl)
    return labels, lbl

# Function to relabel data based on the label mapping
def relabel_data_overlap_new(data1, data2, label_mapping, data1_unique, data2_unique):
    relabeled_data = np.zeros_like(data2)
    label_mapping = {k: label_mapping[k] for k in sorted(label_mapping, reverse=True)}
    
    for dummy_label, real_label in label_mapping.items():
        union_points=set()
        for lbl in [dummy_label, real_label]:
            set1 = set(); set2 = set()
            
            if lbl in data1_unique:
                data1_points = np.argwhere(data1 == lbl)
                set1 = set(map(tuple, data1_points)); del data1_points
            
            if lbl in data2_unique:          
                data2_points = np.argwhere(data2 == lbl)
                set2 = set(map(tuple, data2_points)); del data2_points
        
            set_union = set1.union(set2)

            # Perform the union operation
            union_points = union_points.union(set_union)
            del set1; del set2
            del set_union
        
        for coord in union_points:
            # Relabel these points with the new label
            relabeled_data[coord] = real_label
        del union_points
        
    return relabeled_data

# Function to relabel data based on the label mapping
def relabel_data_overlap(data1, data2, label_mapping):
    relabeled_data = np.zeros_like(data2)
    label_mapping = {k: label_mapping[k] for k in sorted(label_mapping, reverse=True)}
    
    for dummy_label, real_label in label_mapping.items():
        union_points=set()
        for lbl in [dummy_label, real_label]:
            data1_points = np.argwhere(data1 == lbl)
            data2_points = np.argwhere(data2 == lbl)

            set1 = set(map(tuple, data1_points))
            set2 = set(map(tuple, data2_points))
            set_union = set1.union(set2)

            # Perform the union operation
            union_points = union_points.union(set_union)
            del set1; del set2
            del set_union #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        for coord in union_points:
            # Relabel these points with the new label
            relabeled_data[coord] = real_label
        del union_points
        
    return relabeled_data

def relabel_data_window(data, label_mapping):
    existing_lbls=list(np.unique(data))
    if 0 in existing_lbls: existing_lbls.remove(0)
    
    for dummy_label, real_label in label_mapping.items():
        if dummy_label != real_label and dummy_label in existing_lbls:
            data[data == dummy_label] = real_label
    return data

def relabel_data_window_xarray(data: xr.DataArray, label_mapping: dict) -> xr.DataArray:
    # Perform the relabeling based on the label_mapping dictionary
    for dummy_label, real_label in label_mapping.items():
        if dummy_label != real_label:
            data = xr.where(data == dummy_label, real_label, data)
    return data

def update_overlap(previous_overlap, current_overlap):
    label_mapping, data1_unique, data2_unique = create_label_pairing_new(previous_overlap, current_overlap)
    overlap_window = relabel_data_overlap_new(previous_overlap, current_overlap, label_mapping, data1_unique, data2_unique)    
    return overlap_window, label_mapping

def concatenate_nc_files(existing_nc_folder, existing_nc_filename, new_dataset):
    # Construct the path to the existing NetCDF file
    existing_nc_path = f"{existing_nc_folder}/{existing_nc_filename}"
    
    # Open the existing NetCDF file using xarray
    existing_dataset = xr.open_dataset(existing_nc_path)
    
    # Ensure that both datasets have the same lat/lon dimensions before concatenation
    if not (existing_dataset.lat.equals(new_dataset.lat) and existing_dataset.lon.equals(new_dataset.lon)):
        raise ValueError("The latitude and longitude dimensions of the datasets do not match.")
    
    # Concatenate along the time dimension
    concatenated_dataset = xr.concat([existing_dataset, new_dataset], dim='time')
    
    return concatenated_dataset

def remove_all_files_in_folder(folder_path):
    # Loop through each file in the directory
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Check if it is a file and not a directory
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                print(f"Skipped directory: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
            
def get_files_with_substring(folder_path, substring):
    # List to store the file names
    file_list = []

    # Loop through each file in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file name contains the substring
        if substring in file_name:
            file_list.append(file_name)
    
    return file_list

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
    
# Function to save video from 2D frames
def save_video(frames, output_file, fps=5):
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        labels = list(np.unique(frame))
        if 0 in labels: labels.remove(0)
        frame_aux=frame
        
        # Ensure the frame is uint8 and in the range [0, 255]
        frame = np.float64(frame); frame /= np.max(frame)
        frame *= 255; frame = np.uint8(frame)

        # Convert to 3-channel
        frame_3d = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(frame_3d, str(i), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

        # Loop over each label and draw the label at the centroid
        for label in labels:  # Start from 1 to skip the background
            indices = np.argwhere(frame_aux == label)
            centroid = indices.mean(axis=0)
            x_centroid, y_centroid = int(centroid[1]), int(centroid[0])
            label_text = str(label)

            # Draw the label text at the centroid
            cv2.putText(frame_3d, label_text, (x_centroid, y_centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

        out.write(frame_3d)

    out.release()
    cv2.destroyAllWindows()
    
def cantor_pairing(k1, k2):
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

####################################    for multiprocessing        #########################
import multiprocessing as mp
def process_chunk_cantor(chunk):
    frame1, frame2 = chunk
    return np.vectorize(cantor_pairing)(frame1, frame2)

def parallel_cantor_pairing(frame1, frame2, num_processes):
    chunks = np.array_split(frame1, num_processes), np.array_split(frame2, num_processes)
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk_cantor, zip(*chunks))
    
    paired_array = np.concatenate(results, axis=0)
    return paired_array

def process_chunk_counter(chunk):
    return Counter(chunk)

def parallel_counter(frames, num_processes):
    chunks = np.array_split(frames, num_processes)
    
    with mp.Pool(processes=num_processes) as pool:
        counter_chunks = pool.map(process_chunk_counter, chunks)
    
    # Merge counters
    combined_counter = Counter()
    for counter in counter_chunks:
        combined_counter.update(counter)
    
    return combined_counter

def replace_labels(chunk, curr_lbl, update_lbl):
    chunk[chunk == curr_lbl] = update_lbl
    return chunk

def parallel_replace_labels(curr_frame, curr_lbl, update_lbl, num_processes):
    chunks = np.array_split(curr_frame, num_processes)
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(replace_labels, [(chunk, curr_lbl, update_lbl) for chunk in chunks])
    
    return np.concatenate(results, axis=0)

#   for update_frame2d
def update_label_for_curr_lbl(curr_lbl, count_dict, inv_unique_codes, unique_codes, curr_frame):
    update_lbl = curr_lbl
    best_match_count = 0
    
    curr_frame[curr_frame != curr_lbl]=0
    
    for lbl_tuple, code in unique_codes.items():
        if lbl_tuple[1] == curr_lbl and lbl_tuple[0] not in [0, curr_lbl]:  # only search for other lbl's matches with this curr_lbl
        
            overlay_count = get_overlay_count(count_dict, inv_unique_codes, lbl_tuple) > 0.5

            if count_dict[code] > best_match_count and overlay_count > 0 and (count_dict[code] / overlay_count) > 0.5:
                best_match_count = count_dict[code]
                update_lbl = lbl_tuple[0]

    remove_label=None; add_label=None
    if update_lbl != curr_lbl:
        curr_frame[curr_frame == curr_lbl] = update_lbl
        remove_label = curr_lbl
        add_label = update_lbl

    return curr_frame, remove_label, add_label

def parallel_modify(args_unfolded):
    n_processes=8
    curr_frame=args_unfolded[-1]
    new_labels_aux=args_unfolded[0]
    
    # Define the function to be parallelized (modify_array) and its arguments
    args=list()
    for curr_lbl in new_labels_aux:
        #parameters: new_labels_aux, count_dict, inv_unique_codes, unique_codes, curr_frame
        args.append((curr_lbl, args_unfolded[1].copy(), args_unfolded[2].copy(),
                                args_unfolded[3][curr_lbl].copy(), args_unfolded[4].copy())) #atenção à modificação args_unfolded[3][curr_lbl]
    del args_unfolded
    
    with mp.Pool(processes=n_processes) as pool:
        # Use pool.starmap to apply modify_array in parallel
        results = pool.starmap(update_label_for_curr_lbl, args)
    del args
        
    # Aggregate results by summing up all modified frames
    final_frame = np.zeros_like(curr_frame)
    for result in results:
        frame = result[0]; remove_lbl = result[1]; add_lbl = result[2]
        final_frame += frame
        if remove_lbl and remove_lbl in new_labels_aux: new_labels_aux.remove(remove_lbl)
        if add_lbl and add_lbl not in new_labels_aux: new_labels_aux.append(add_lbl)
    
    if 0 not in new_labels_aux: new_labels_aux=[0]+new_labels_aux
    return final_frame, new_labels_aux

#dict of dictionairies. first keys are each curr label in unique codes. for each curr label, we have the lable tuples that fit the conditions for overlap check.
#this is only made to speed up the process. Instead of always doing a for cycle to find the fit lbl tuples, we run only once to have a dictionary which directly access to the matches
def unique_codes_to_dict(unique_codes):
    unique_codes_dict=dict(); visited_lbls=list()
    for lbl_tuple, code in unique_codes.items():
        #if lbl_tuple[1] in labels and lbl_tuple[0] not in [0, lbl_tuple[1]]: #AQUI É lbl_tuple[1]? COMO CONTROLAR ISTO NESTE FORMATO "in labels"???
        #if lbl_tuple[1] == curr_lbl and lbl_tuple[0] not in [0, curr_lbl]:
        if lbl_tuple[1]==24:
            print('stop here')
        
        if lbl_tuple[1] in visited_lbls:#list(unique_codes_dict.keys()):
            unique_codes_dict[lbl_tuple[1]][lbl_tuple]=code
        else:
            unique_codes_dict[lbl_tuple[1]] = {lbl_tuple: code} 
            visited_lbls.append(lbl_tuple[1]) 
    return unique_codes_dict

def update_frame2d_mp(curr_frame, prev_frame, prev_unique):
    start_times={}; end_times={}
    start_times['runtime']=time.time()
    
    #uncomment for old version
    #new_labels = list(np.unique(curr_frame))
    #unique_codes, inv_unique_codes = get_unique_code(prev_unique, new_labels)
    #count_dict, _ = get_pairing_count(prev_frame, curr_frame, prev_unique, new_labels, unique_codes, only_count_dict=True)
    count_dict, inv_unique_codes, unique_codes, new_labels = get_pairing_count(prev_frame, curr_frame)
    
    new_labels_aux = new_labels.copy()
    if 0 in new_labels_aux: new_labels_aux.remove(0)

    unique_codes_dict = unique_codes_to_dict(unique_codes)
    
    MIN_N_LABELS=-1
    if len(new_labels_aux)>MIN_N_LABELS:
        args_unfolded=[new_labels_aux, count_dict, inv_unique_codes, unique_codes_dict, curr_frame]
        curr_frame, new_labels = parallel_modify(args_unfolded)
    else:
        for curr_lbl in new_labels_aux:
            update_lbl = curr_lbl
            best_match_count=0
            
            for lbl_tuple, code in unique_codes.items():
                if lbl_tuple[1] == curr_lbl and lbl_tuple[0] not in [0, curr_lbl]: #only search for other lbl's matches with this curr_lbl

                    overlay_count = get_overlay_count(count_dict, inv_unique_codes, lbl_tuple)>0.5 #frame1_count[lbl_tuple[0]]>0.5 

                    if count_dict[code] > best_match_count and overlay_count>0 and (count_dict[code]/overlay_count) >0.5:
                        best_match_count = count_dict[code] 
                        update_lbl = lbl_tuple[0]
            
            if update_lbl != curr_lbl:
                curr_frame[curr_frame==curr_lbl] = update_lbl
                new_labels.remove(curr_lbl)
                new_labels.append(update_lbl)
    end_times['runtime']=time.time()
    runtime_steps(start_times, end_times, f'update_frame ({len(new_labels_aux)} labels): ')

    return curr_frame, new_labels

##########################################################################################

def get_unique_code(labels1, labels2):
    # Generate unique codes for all combinations
    unique_codes = {}
    inv_unique_codes = {}
    for num1 in labels1:
        for num2 in labels2:
            unique_code = cantor_pairing(num1, num2)
            unique_codes[(num1, num2)] = unique_code
            inv_unique_codes[unique_code] = (num1, num2)
    return unique_codes, inv_unique_codes

def get_pairing_count(frame1, frame2):
    paired_array = np.vectorize(cantor_pairing)(frame1, frame2)
    #paired_array = parallel_cantor_pairing(frame1, frame2, num_processes=6)
    
    flat_paired = paired_array.flatten()
    count_dict = Counter(flat_paired); del flat_paired
    #count_dict = parallel_counter(flat_paired, num_processes=6); del flat_paired
    existing_codes = list(count_dict.keys())
    
    new_labels=list()
    unique_codes=dict(); inv_unique_codes=dict()
    for code in existing_codes:
        aux_tuple = decode_cantor(code)
        unique_codes[aux_tuple] = code
        inv_unique_codes[code] = aux_tuple
        if aux_tuple[1] not in new_labels: new_labels.append(aux_tuple[1])
    del existing_codes

    return count_dict, inv_unique_codes, unique_codes, new_labels

def get_pairing_count_old(frame1, frame2, labels1, labels2, unique_codes, only_count_dict=False):
    paired_array = np.vectorize(cantor_pairing)(frame1, frame2)
    #paired_array = parallel_cantor_pairing(frame1, frame2, num_processes=6)
    
    flat_paired = paired_array.flatten()
    count_dict = Counter(flat_paired); del flat_paired
    #count_dict = parallel_counter(flat_paired, num_processes=6); del flat_paired
    
    if only_count_dict:
        return count_dict, paired_array
    else:
        label_match = dict(); total_overlap = dict(); frame1_count = dict(); frame2_count = dict()
        for lbl in labels1:
            if lbl != 0:
                if lbl not in labels2: #if lbl  in 1 not in 2, it must be closed
                    label_match[lbl]=0
                    total_overlap[lbl] = 0
                else: #if lbl 1 in 2, get the count of times they are paired
                    code = unique_codes[(lbl,lbl)] #get the cantor code
                    label_match[lbl] = count_dict[code] # with the code, get the count of times it happened in paired_array
                    
                    #calculate total overlap for that area
                    for lbl_tuple, code in unique_codes.items():
                        if lbl_tuple[0] == lbl or lbl_tuple[1] == lbl:
                            if lbl in total_overlap.keys(): total_overlap[lbl] += count_dict[code]
                            else: total_overlap[lbl] = count_dict[code]
                        
                        #count lbl's frame1 pixels
                        if lbl_tuple[0] == lbl:
                            if lbl in frame1_count.keys(): frame1_count[lbl] += count_dict[code]
                            else: frame1_count[lbl] = count_dict[code]
                        
                        #count lbl's frame2 pixels
                        if lbl_tuple[1] == lbl:
                            if lbl in frame2_count.keys(): frame2_count[lbl] += count_dict[code]
                            else: frame2_count[lbl] = count_dict[code]
                            
        return label_match, total_overlap, frame1_count, frame2_count, count_dict

def get_colors(last_lbl):
    #plot colors just for debugging
    n_labels = last_lbl+40
    colors = plt.cm.viridis(np.linspace(0, 1, n_labels))  # You can choose any colormap you like
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_labels+0.5, 1), ncolors=n_labels)
    return cmap, norm

def plot_mhw(frame, txt, cmap, norm, with_text=True):
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
    
def get_overlay_count(count_dict, inv_unique_codes, lbl_tuple):    
    overlay_count=0
    for code, counting in dict(count_dict).items():
        if code in inv_unique_codes.keys():
            tup = inv_unique_codes[code]
            if tup[0] == lbl_tuple[0] or tup[1] == lbl_tuple[1]:
                overlay_count+=counting
    return overlay_count


def update_frame2d(curr_frame, prev_frame, prev_unique):
    start_times={}; end_times={}
    start_times['unique_codes']=time.time()
    new_labels = list(np.unique(curr_frame))
    unique_codes, inv_unique_codes = get_unique_code(prev_unique, new_labels)
    end_times['unique_codes']=time.time()
    
    start_times['pairing_count']=time.time()
    count_dict, _ = get_pairing_count(prev_frame, curr_frame, prev_unique, new_labels, unique_codes, only_count_dict=True)
    end_times['pairing_count']=time.time()
    
    start_times['overlay_loop']=time.time()
    new_labels_aux = new_labels.copy(); new_labels_aux.remove(0)
    for curr_lbl in new_labels_aux:
        update_lbl = curr_lbl
        best_match_count=0
        
        for lbl_tuple, code in unique_codes.items():
            if lbl_tuple[1] == curr_lbl and lbl_tuple[0] not in [0, curr_lbl]: #only search for other lbl's matches with this curr_lbl

                overlay_count = get_overlay_count(count_dict, inv_unique_codes, lbl_tuple)>0.5 #frame1_count[lbl_tuple[0]]>0.5 

                if count_dict[code] > best_match_count and overlay_count>0 and (count_dict[code]/overlay_count) >0.5:
                    best_match_count = count_dict[code] 
                    update_lbl = lbl_tuple[0]
        
        if update_lbl != curr_lbl:
            curr_frame[curr_frame==curr_lbl] = update_lbl
            #curr_frame=parallel_replace_labels(curr_frame, curr_lbl, update_lbl, num_processes=4)
            new_labels.remove(curr_lbl)
            new_labels.append(update_lbl)
    end_times['overlay_loop']=time.time()
    runtime_steps(start_times, end_times, ' update_frame: ')
       
    return curr_frame, new_labels


def detect_frame_by_frame(array3d, params, verbose=False):
    labels = list()
    last_lbl=1
    
    structure = params.getp('neighbours')
    
    prev_frame = array3d[0]
    prev_frame, _ = scp.label(prev_frame, structure)
    labels.append(prev_frame)
    prev_unique = list(np.unique(prev_frame))
    last_lbl = max(last_lbl, max(prev_unique)+1)
    if verbose:
        cmap, norm = get_colors(last_lbl)
        plot_mhw(prev_frame, f'frame0', cmap, norm)
    
    n_frames=len(array3d)-1
    for idx, curr_frame in enumerate(array3d[1:]):
        print(f'frame {idx+1}/{n_frames}')
        curr_frame, _ = scp.label(curr_frame, structure)
        curr_frame, last_lbl = edit_labels_faster(curr_frame, last_lbl)
        
        if verbose: plot_mhw(curr_frame, f'frame{idx+1} before update', cmap, norm)
        curr_frame, prev_unique = update_frame2d_mp(curr_frame, prev_frame, prev_unique)
        if verbose: plot_mhw(curr_frame, f'frame{idx+1} after update', cmap, norm, with_text=True)

        labels.append(curr_frame)
        prev_frame = curr_frame
    
    del array3d
    
    labels = np.array(labels)
    labels = cc3d.dust( labels, threshold = params.getp('min_pixels'), connectivity=6, in_place=False)
    return labels

def extract_year_from_filename(filename):
    # This regex looks for a 4-digit number in the filename, which we assume is the year
    match = re.search(r'\b(19|20)\d{2}\b', filename)
    if match: return int(match.group())
    return None

def get_most_recent_nc_file(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter out non-netCDF files
    nc_files = [file for file in files if file.endswith('.nc')]
    
    # Extract years and find the most recent file
    most_recent_file = None; most_recent_year = -1
    for file in nc_files:
        year = extract_year_from_filename(file)
        if year and year > most_recent_year:
            most_recent_year = year; most_recent_file = file
    return most_recent_file

def open_most_recent_nc_file(folder_path):
    most_recent_file = get_most_recent_nc_file(folder_path)
    if most_recent_file:
        file_path = os.path.join(folder_path, most_recent_file)
        ds = xr.open_dataset(file_path)
        return ds
    else:
        print("No valid netCDF files found in the specified folder.")
        return None

#start mhw detection process in the middle of the intervals instad of starting over again by assigning the correct overlap window and the date stamps
def begin_midway(ds, window_size, step_size):
    overlap_time = pd.to_datetime(ds['time'].values).to_pydatetime()
    overlap_size = window_size-step_size

    overlap_end = overlap_time[-1]
    overlap_start = overlap_end - overlap_size

    window_start_day = overlap_start
    window_end_day = overlap_start + window_size

    previous_overlap = ds.sel(time = slice(overlap_start, overlap_end))
    
    return window_start_day, window_end_day, previous_overlap, overlap_start, overlap_end

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