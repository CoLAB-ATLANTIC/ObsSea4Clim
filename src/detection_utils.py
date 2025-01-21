import numpy as np
import os
import cc3d
import math
import time
import pickle
import scipy.ndimage as scp
import multiprocessing as mp
from collections import Counter
from tqdm import tqdm

import src.utils as utils
import src.config as config
from src.config import FLAGS

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

def get_overlay_count(count_dict, inv_unique_codes, lbl_tuple):    
    overlay_count=0
    for code, counting in dict(count_dict).items():
        if code in inv_unique_codes.keys():
            tup = inv_unique_codes[code]
            if tup[0] == lbl_tuple[0] or tup[1] == lbl_tuple[1]:
                overlay_count+=counting
    return overlay_count

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
    
    flat_paired = paired_array.flatten()
    count_dict = Counter(flat_paired); del flat_paired
    existing_codes = list(count_dict.keys())
    
    new_labels=list()
    unique_codes=dict(); inv_unique_codes=dict()
    for code in existing_codes:
        aux_tuple = decode_cantor(code)
        unique_codes[aux_tuple] = code
        inv_unique_codes[code] = aux_tuple
        if aux_tuple[1] not in new_labels: new_labels.append(aux_tuple[1])

    return count_dict, inv_unique_codes, unique_codes, new_labels

# dict of dictionairies. first keys are each curr label in unique codes.\\
#  for each curr label, we have the lable tuples that fit the conditions \\
# for overlap check.
# this is only made to speed up the process. Instead of always doing a for\\
#  cycle to find the fit lbl tuples, we run only once to have a dictionary\\
#  which directly access to the matches
def unique_codes_to_dict(unique_codes):
    unique_codes_dict=dict(); visited_lbls=list()
    for lbl_tuple, code in unique_codes.items():
        
        if lbl_tuple[1] in visited_lbls:
            unique_codes_dict[lbl_tuple[1]][lbl_tuple]=code
        else:
            unique_codes_dict[lbl_tuple[1]] = {lbl_tuple: code} 
            visited_lbls.append(lbl_tuple[1]) 
    return unique_codes_dict

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
    N_PROCESSES=8
    curr_frame = args_unfolded[-1]
    new_labels_aux = args_unfolded[0]
    
    # Define the function to be parallelized (modify_array) and its arguments
    args=list()
    for curr_lbl in new_labels_aux:
        #parameters: new_labels_aux, count_dict, inv_unique_codes, unique_codes, curr_frame
        args.append((curr_lbl, args_unfolded[1].copy(), args_unfolded[2].copy(),
                                args_unfolded[3][curr_lbl].copy(), args_unfolded[4].copy()))
    del args_unfolded
    
    with mp.Pool(processes=N_PROCESSES) as pool:
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

def update_frame2d_mp(curr_frame, prev_frame, verbose = False):
    start_times={}; end_times={}
    start_times['runtime']=time.time()
    
    count_dict, inv_unique_codes, unique_codes, new_labels = get_pairing_count(prev_frame, curr_frame)
    
    new_labels_aux = new_labels.copy()
    if 0 in new_labels_aux: new_labels_aux.remove(0)

    unique_codes_dict = unique_codes_to_dict(unique_codes)
    
    MIN_N_LABELS = -1
    if len(new_labels_aux) > MIN_N_LABELS:
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
    if verbose: utils.runtime_steps(start_times, end_times, f'update_frame ({len(new_labels_aux)} labels): ')

    return curr_frame, new_labels

def detect_frame_by_frame(array3d, verbose=False):
    labels = list()
    last_lbl=1
    
    structure = config.NEIGHBOURHOOD

    min_pixels_frame = calculate_num_pixels(FLAGS.min_area_frame, config.KM_RESOLUTION)
    min_pixels_time = calculate_num_pixels(FLAGS.min_area_time, config.KM_RESOLUTION)
    min_pixels_frame = int(min_pixels_frame/(config.DOWNSAMPLE_RATIO**2))
    min_pixels_time = int(min_pixels_time/(config.DOWNSAMPLE_RATIO**2))
    #squared or cubed???


    prev_frame = array3d[0]
    prev_frame, _ = scp.label(prev_frame, structure)
    prev_frame = cc3d.dust(prev_frame, threshold = min_pixels_frame, connectivity=config.CONNECTIVITY, in_place=True)
    labels.append(prev_frame)
    prev_unique = list(np.unique(prev_frame))

    last_lbl = max(last_lbl, max(prev_unique)+1)
    if verbose:
        cmap, norm = utils.get_colors(last_lbl)
        utils.plot_frame(prev_frame, f'frame0', cmap, norm)
    
    #n_frames=len(array3d)-1
    for idx, curr_frame in enumerate(tqdm(array3d[1:], desc="Processing frames")):
        #print(f'frame {idx+1}/{n_frames}')
        curr_frame, _ = scp.label(curr_frame, structure)
        curr_frame = cc3d.dust( curr_frame, threshold = min_pixels_frame, connectivity=config.CONNECTIVITY, in_place=True)
        curr_frame, last_lbl = edit_labels_faster(curr_frame, last_lbl)
        
        if verbose: utils.plot_frame(curr_frame, f'frame{idx+1} before update', cmap, norm)
        curr_frame, prev_unique = update_frame2d_mp(curr_frame, prev_frame)
        if verbose: utils.plot_frame(curr_frame, f'frame{idx+1} after update', cmap, norm, with_text=True)

        labels.append(curr_frame)
        prev_frame = curr_frame
    
    del array3d
    
    labels = np.array(labels)
    labels = cc3d.dust( labels, threshold = min_pixels_time, connectivity=config.CONNECTIVITY, in_place=True)
    return labels

def detect_mhws_2d(array3d, lbl):
    labels = detect_frame_by_frame(array3d, verbose= False)
    labels = labels.astype('uint16')
    labels, lbl = edit_labels(labels, lbl)
    return labels, lbl

def cantor_pairing(k1, k2):
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

def decode_cantor(z):
    # Compute w
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    # Compute t
    t = w * (w + 1) // 2
    # Compute y and x
    y = z - t
    x = w - y
    return (int(x), int(y))

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
    if os.path.exists(config.MAPPING_FILE):
        with open(config.MAPPING_FILE, 'rb') as f:
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
                    
        with open( config.MAPPING_FILE, 'wb') as outf:
            pickle.dump(label_mapping, outf)
        
    return label_mapping

def create_label_pairing(data1, data2):
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

# Function to relabel data based on the label mapping
def relabel_data_overlap(data1, data2, label_mapping, data1_unique, data2_unique):
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

def update_overlap(previous_overlap, current_overlap):
    label_mapping, data1_unique, data2_unique = create_label_pairing(previous_overlap, current_overlap)
    overlap_window = relabel_data_overlap(previous_overlap, current_overlap, label_mapping, data1_unique, data2_unique)    
    return overlap_window, label_mapping

def relabel_data_window(data, label_mapping):
    existing_lbls=list(np.unique(data))
    if 0 in existing_lbls: existing_lbls.remove(0)
    
    for dummy_label, real_label in label_mapping.items():
        if dummy_label != real_label and dummy_label in existing_lbls:
            data[data == dummy_label] = real_label
    return data


def calculate_num_pixels(total_area, km_res):
    """
    Calculate the number of pixels needed to cover a given area.

    Parameters:
    - total_area (float): The total desired area in square kilometers.
    - km_res (float): The resolution of each pixel in kilometers.

    Returns:
    - num_pixels (int): The required number of pixels.
    """
    # Area of each pixel
    pixel_area = km_res ** 2

    # Calculate the number of pixels
    num_pixels = total_area / pixel_area

    # Return the number of pixels as an integer
    return int(np.ceil(num_pixels))  # Round up to ensure coverage