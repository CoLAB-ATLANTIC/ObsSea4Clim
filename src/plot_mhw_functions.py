import imports_and_functions as impf

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
import re
warnings.filterwarnings('ignore')
import xarray as xr
import os
from scipy.ndimage import measurements
from datetime import datetime, timedelta
import pandas as pd

class VariableStorage:
    def __init__(self, mhw_zones_folder=None, mhw_intensity_folder=None, mask_fullpath=None,
                 longhurst_path=None, output_video_path=None, downsample_ratio=None, fps=None, ID=None, verbose= False):
        self.mhw_zones_folder = mhw_zones_folder
        self.mhw_intensity_folder = mhw_intensity_folder
        self.mask_fullpath = mask_fullpath
        self.longhurst_path = longhurst_path
        self.output_video_path = output_video_path
        
        self.downsample_ratio = downsample_ratio
        self.fps = fps
        
        self.ID=ID
        
        self.verbose=verbose

def get_nc_files(vars):
    mhw = xr.open_dataset(vars.mhw_zones_folder+f'{vars.ID}.nc')

    time = mhw.time.values
    first_year = impf.to_datetime(time[0]).year
    last_year = impf.to_datetime(time[-1]).year
    del time

    intensity_files = [vars.mhw_intensity_folder + f'mhw_{year}.nc' for year in range(first_year, last_year+1)]
    intensity_nc = xr.open_mfdataset(intensity_files)
    
    mhw = impf.downsample_netcdf(mhw, ratio=vars.downsample_ratio)
    intensity_nc = impf.downsample_netcdf(intensity_nc, ratio=vars.downsample_ratio)
    if vars.verbose: print(f'(lat, lon) size: ({len(mhw.lat.values)}, {len(mhw.lon.values)})')
    
    return mhw, intensity_nc

def filter_files_by_year(file_list, start_year, end_year):
    filtered_files = [filename for filename in file_list if start_year <= extract_year(filename) <= end_year]
    return filtered_files

# Function to extract the year from the filename
def extract_year(filename):
    match = re.search(r'_(\d{4})_[\w-]+_', filename)
    if match:
        return int(match.group(1))
    return -1  # In case no year is found

# Function to extract the serial number from the filename
def extract_serial_number(filename):
    match = re.search(r'_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1  # In case no serial number is found

def get_mask_and_longhurst(vars, lat):
    mask = xr.open_dataset(vars.mask_fullpath)
    mask = impf.downsample_netcdf(mask, ratio=vars.downsample_ratio)
    mask = mask.sel(lat=slice(lat.min(), lat.max()))
    land_flag = 2
    mask = xr.where(mask == land_flag, np.nan, 1)
    mask_values = mask.mask.values

    longhurst = impf.gpd.read_file(vars.longhurst_path)
    return mask_values, longhurst

def get_color_by_label(labels):
    # Define unique labels including zero
    unique_labels = np.array([0] + labels)

    # Create a colormap with white for zero and specific colors for each label
    num_labels = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, num_labels - 1))  # Use the 'viridis' colormap excluding zero
    colors = np.vstack(([1, 1, 1, 1], colors))  # Add white color for zero at the beginning
    cmap = ListedColormap(colors)

    # Create boundaries for norm: mid-points between labels
    boundaries = np.concatenate(([unique_labels[0] - 0.5],
                                 unique_labels[:-1] + 0.5,
                                 [unique_labels[-1] + 0.5]))
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, boundaries

def get_colormaps(labels=None):
    # Define the colors for the colormap
    colors = [(1, 1, 1), (0.45, 0.7, 0.85),
            (0.65, 0.85, 0.75), (0.8, 0.925, 0.75),
            (0.95, 0.98, 0.7), (0.99, 0.9, 0.55),
            (0.99, 0.8, 0.45), (0.99, 0.6, 0.45),
            (0.99, 0.4, 0.45), (0.95, 0.2, 0.45)]
    # Create the colormap
    cmap_intensity = LinearSegmentedColormap.from_list('custom_white_to_yellow_to_red', colors)

    # Create a colormap with n_labels unique colors
    if labels:
        """ n_labels=len(labels)
        colors = plt.cm.get_cmap('viridis', n_labels)
        newcolors = colors(np.linspace(0, 1, n_labels))
        newcolors[0] = np.array([1, 1, 1, 1])  # Set first color (corresponding to zero) to white
        cmap_zones = ListedColormap(newcolors) """
        cmap_zones, norm, boundaries = get_color_by_label(labels)
        
        return cmap_intensity, cmap_zones, norm, boundaries
    else:
        #just one label
        colors = np.array([[1, 1, 1, 1],  # White for 0
                        [0.8, 0, 0, 1]]) # Red for 1
        binary_cmap = ListedColormap(colors)
        return cmap_intensity, binary_cmap, None, None

from matplotlib.patheffects import withStroke
def add_centroid_labels(ax, data, labels, IDs, lon, lat):
    """
    Add text labels to the centroid of each label region in the data.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on.
    data (numpy.ndarray): The data array containing labels.
    labels (list): The list of unique labels to annotate.
    lon (numpy.ndarray): The longitude values for the x-axis.
    lat (numpy.ndarray): The latitude values for the y-axis.
    """
    for label in labels:
        mask = data == label
        if mask.any():
            y, x = measurements.center_of_mass(mask)
            lon_center = lon[int(round(x))]
            lat_center = lat[int(round(y))]
            #ax.text(lon_center, lat_center, IDs[label], color='black', ha='center', va='center', fontsize=12, weight='bold')
    
            # Define text properties
            text_props = dict(color='white', ha='center', va='center', fontsize=16, weight='bold')
            
            # Define the outline effect
            outline_effect = [withStroke(linewidth=3, foreground='black')]
            
            # Add text with outline effect
            ax.text(lon_center, lat_center, IDs[label], path_effects=outline_effect, **text_props)

def create_plots(vars, time, mhw_data, intensity_nc, mask_values, longhurst, lat, lon, labels=None, IDs=None):
    
    cmap_intensity, binary_cmap, norm, boundaries = get_colormaps(labels)
    
    figs=list()
    i=1
    for date, mhw_frame in zip(time, mhw_data):
        if vars.verbose: print(f'Rendering frame {i}/{len(time)}'); i+=1
        
        intensity_data = intensity_nc.sel(time=date)
        intensity_frame = intensity_data.intensity.values
        intensity_frame *= mask_values
        
        mhw_frame = mhw_frame*mask_values
        
        fig, axs = plt.subplots(1, 2, figsize=(40, 10))
        latlon_fontsize = 14
        subplot_fontsize = 18
        subtitle_fs = 24
        colorbar_fs =14
        
        contour1 = axs[0].contourf(lon, lat, intensity_frame, cmap=cmap_intensity,
                                levels=np.arange(0, 10, 0.1))
        
        axs[0].set_xlabel('Lon', fontsize=latlon_fontsize); axs[0].set_ylabel('Lat', fontsize=latlon_fontsize)
        axs[0].set_xlim(-90, 10); axs[0].set_ylim(10, 60)
        axs[0].set_title('MHW Intensity Input', fontsize=subplot_fontsize)
        longhurst.plot(ax=axs[0], edgecolor='black', facecolor='none', linewidth=0.5)
        axs[0].set_aspect('equal')
        axs[0].set_facecolor("black")

        # Add colorbar for the first subplot
        divider = make_axes_locatable(axs[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.1) 
        cbar1 = fig.colorbar(contour1, ax=axs[0], cax=cax1)
        cbar1.set_label('$\Delta$K', fontsize=colorbar_fs)
        
        ##########   Figure2    #########################################################
        if norm:
            contour2 = axs[1].contourf(lon, lat, mhw_frame, levels=boundaries, cmap=binary_cmap, norm=norm)
            add_centroid_labels(axs[1], mhw_frame, labels, IDs, lon, lat)
        else: contour2 = axs[1].contourf(lon, lat, mhw_frame, cmap=binary_cmap)
        
        axs[1].set_xlabel('Lon', fontsize=latlon_fontsize); axs[1].set_ylabel('Lat', fontsize=latlon_fontsize)
        axs[1].set_xlim(-90, 10); axs[1].set_ylim(10, 60)
        axs[1].set_title('Detected MHW Zones', fontsize=subplot_fontsize)
        longhurst.plot(ax=axs[1], edgecolor='black', facecolor='none', linewidth=0.5)
        axs[1].set_aspect('equal')
        axs[1].set_facecolor("black")
        
        date_str = impf.to_datetime(date).strftime('%Y-%m-%d')
        fig.suptitle(f'Detected MHW: {vars.ID} ; Current Date: {date_str}', fontsize=subtitle_fs)
        
        plt.subplots_adjust(wspace=-0.1, hspace=0.0)
        plt.tight_layout()
        #plt.show()
        figs.append(fig)
        plt.close(fig) 
        
    return figs

def create_video_frames(figs):
    frames = []
    for fig in figs:
        canvas = FigureCanvas(fig)
        canvas.draw()  # Draw the figure
        img = np.array(canvas.renderer.buffer_rgba())  # Convert to RGBA buffer
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert to BGR (OpenCV) format
        frames.append(img_bgr)
        plt.close(fig)  # Close the figure to free memory
    return frames

def render_video(vars, frames):
    # Parameters
    width, height = frames[0].shape[1], frames[0].shape[0]  # Use the size of the first frame

    # Create VideoWriter object
    out = cv2.VideoWriter(vars.output_video_path+f'{vars.ID}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vars.fps, (width, height))

    # Write frames to video
    for frame in frames:
        out.write(frame)

    # Release VideoWriter
    out.release()
    
def create_one_mhw_video(vars, labels=None, IDs=None):
    mhw, intensity_nc = get_nc_files(vars)
    
    mhw_data = mhw.mhw_zone.values
    time = mhw.time.values
    lat = mhw.lat.values; lon = mhw.lon.values
    mhw.close()
    
    mask_values, longhurst = get_mask_and_longhurst(vars, lat)
    figs = create_plots(vars, time, mhw_data, intensity_nc,
                        mask_values, longhurst, lat, lon, labels, IDs)
    del mhw_data; del time; del lat; del lon
    intensity_nc.close()
    
    frames = create_video_frames(figs); del figs
    render_video(vars, frames); del frames
    
    
############################################################################
##  functions for the video creation by time period
############################################################################

def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name

# Function to extract the serial number from the filename
def extract_serial_number(filename):
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return -1  # In case no serial number is found

def load_and_filter_nc_files(file_paths, start_date, end_date):
    datasets = []
    labels=[]
    IDs = dict()
    for file_path in file_paths:
        ds = xr.open_dataset(file_path)
        filtered_ds = ds.sel(time=slice(start_date, end_date))
        if not filtered_ds.time.size == 0:
            lbl=extract_serial_number(file_path)
            #multiply data by the serial number so that each mhw has a different label
            filtered_ds['mhw_zone'] = filtered_ds['mhw_zone'] * lbl
            datasets.append(filtered_ds)
            labels.append(lbl)
            IDs[lbl] = get_filename_without_extension(file_path)
    ds.close(); filtered_ds.close()
    labels.sort()
    return datasets, labels, IDs

def sum_nc_datasets(datasets):
    # Ensure all datasets are aligned on the same time coordinates
    combined = xr.concat(datasets, dim="time")
    summed = combined.groupby("time").sum(dim="time"); combined.close()
    return summed

def create_zero_filled_netcdf(ds_orig, start_date, end_date):
    start_date=datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date=datetime.strptime(end_date, '%Y-%m-%d').date()
    # Determine the dimensions from the original dataset
    lat_dim = len(ds_orig['lat']); lon_dim = len(ds_orig['lon'])

    # Extend time dimension with new dates
    new_time_dim = (end_date - start_date).days + 1
    times = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a new dataset with the same lat, lon structure
    ds_new = xr.Dataset(coords={'lat': ds_orig['lat'], 'lon': ds_orig['lon']})
    # Add time coordinate and fill with new times
    ds_new['time'] = times

    # Create a new DataArray filled with zeros
    data_zeros_shape = (new_time_dim, lat_dim, lon_dim)
    data_zeros = np.zeros(data_zeros_shape)
    # Copy variable attributes
    attrs = ds_orig['mhw_zone'].attrs
    # Create the DataArray for the variable
    da_var = xr.DataArray(data_zeros, dims=('time', 'lat', 'lon'), coords={'time': times, 'lat': ds_orig['lat'], 'lon': ds_orig['lon']}, attrs=attrs)
    # Add the variable to the new dataset
    ds_new['mhw_zone'] = da_var
    
    del data_zeros
    return ds_new