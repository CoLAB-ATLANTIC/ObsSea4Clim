"""
create_video_from_xarray(): Standard function to render a video of an xrarray Dataarray
                            timeseries comprised of (lat,lon,time).

If you want to change plotting parameters, then edit or remake mk1frame() function below.
"""


import cv2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
#from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg

######################################################################
# If you want, make your own "mk1frame()" plotting function :)
######################################################################
def mk1frame(i, dataarray, value_range, title = None, dist_range= False):
    """
    Calculate a value range for a DataArray based on standard deviations
    from the mean.

    Parameters:
        i (int): Timestep index
        dataarray (xarray.DataArray): The input DataArray.
        value_range (list): range of values for colorbar [vmin, mmax]
        title (str): Title of plot. if None inserts variable name
        dist_range (bool): if True, automaticaly adjusts range value
          to data distribution based on standard dev

    Returns:
        fig: figure.
    """

    # assert range values
    if value_range is None:
        if dist_range: vmin, vmax = dist_value_range(dataarray)
        else: vmin, vmax = float(dataarray.min()), float(dataarray.max())
    else: vmin, vmax = value_range

    # create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    dataarray.isel(time=i).plot(ax=ax, vmin=vmin, vmax=vmax,
                                 cbar_kwargs={"shrink": 0.7},
                                 #cmap="coolwarm",
                                 #norm=TwoSlopeNorm(vcenter=0,vmin=vmin, vmax=vmax)
                                    )
    
    if title is None: title = dataarray.name
    date = dataarray.time[i].dt.strftime("%Y-%m-%d").item()
    ax.set_title(f"{title} - {date}")

    return fig
########################################################################


# Functions for creating and rendering video
def render_video(output_video_filepath, frames, fps=5):
    # Parameters
    width, height = frames[0].shape[1], frames[0].shape[0]  # Use the size of the first frame

    # Create VideoWriter object
    out = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Write frames to video
    for frame in frames:
        out.write(frame)

    # Release VideoWriter
    out.release()


# function to transform plot figures into video frames
def create_video_frames(figs):
    frames = []
    for fig in figs:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()  # Draw the figure
        img = np.array(canvas.renderer.buffer_rgba())  # Convert to RGBA buffer
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert to BGR (OpenCV) format
        frames.append(img_bgr)
        plt.close(fig)  # Close the figure to free memory
    return frames


def dist_value_range(dataarray, n_dev=1):
    """
    Calculate a value range for a DataArray based on standard deviations
    from the mean.

    Parameters:
        dataarray (xarray.DataArray): The input DataArray.
        n_dev (int): Number of standard deviations

    Returns:
        tuple: (vmin, vmax) for the colorbar range.
    """
    mean = dataarray.mean(skipna=True).item()
    std = dataarray.std(skipna=True).item()
    vmin = mean - n_dev * std
    vmax = mean + n_dev * std
    return vmin, vmax


def create_video_from_xarray(dataarray, output_video_filepath, value_range=None, title=None, fps=5, dist_range = False):
    """
    Creates a video from an xarray.DataArray over time.

    Parameters:
        dataarray (xarray.DataArray): The input data array containing the time dimension to generate frames.
        output_video_filepath (str): Path to save the output video file. Must have an '.mp4' extension.
        value_range (tuple, optional): A tuple (min, max) defining the color scale value range for the video frames.
          If nan, the min and max values are attributed.
        title (str, optional): Title to display on each frame. If none, the variable name is attributed.
        fps (int, optional): Frames per second for the output video.
        dist_range (bool, optional): If True, adjusts the frame's visualization for a distributed range.
    """

    # Ensure the output directory exists
    output_path = Path(output_video_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate file extension
    if not output_video_filepath.endswith('.mp4'):
        raise ValueError("Output file path must end with '.mp4'")

    # Generate frames for each timestep
    figs = []
    for i in range(len(dataarray.time)):
        fig = mk1frame(i, dataarray, value_range, title, dist_range)
        figs.append(fig)

    # Convert figures to frames
    frames = create_video_frames(figs)
    del figs; del fig

    # Generate frames for each timestep
    render_video(output_video_filepath, frames, fps)
    del frames

'''
# EXAMPLE USAGE:

#open xarray dataset
ds_ehf = xr.open_dataset('/path/to/your/file.nc')
var_name = 'your_var'
#convert to datarray by selecting one variable. Select time range if needed
datarray = ds_ehf[var_name].isel(time=slice(0, 50))

create_video_from_xarray(datarray,
                          output_video_filepath='./xr2video_file.mp4',
                            value_range=None,
                              title=None,
                               fps=5, #bigger values will increase frame rate speed
                                dist_range=False)
'''
