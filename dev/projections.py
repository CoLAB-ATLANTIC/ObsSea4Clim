import numpy as np
import xarray as xr
import xarray as xr
import cartopy.crs as ccrs
from scipy.interpolate import griddata

from multiprocessing import Pool
N_PROCESSORS = 8

def reproject_and_resample_time_step(args):
    """Function to resample a single time step."""
    values, x_coords, y_coords, x_new_grid, y_new_grid = args

    # Flatten the original grid for interpolation
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    var_flat = values.flatten()

    # Interpolate var values to the new grid
    var_resampled = griddata(
        points=(x_flat, y_flat),
        values=var_flat,
        xi=(x_new_grid, y_new_grid),
        method='linear'
    )
    return var_resampled

def reproject_raster(ds, varname, projection=ccrs.PlateCarree(), target_projection=ccrs.Sinusoidal(), cell_size_km=100):
    # Convert cell size in km to projected units (meters)
    cell_size_m = cell_size_km * 1000

    # Extract original latitude and longitude
    latitudes = ds.coords['lat'].values
    longitudes = ds.coords['lon'].values

    # Create a mesh grid of lat/lon
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Transform lat/lon to x/y in the target projection
    points = target_projection.transform_points(projection, lon_grid, lat_grid)
    x_coords = points[:, :, 0]
    y_coords = points[:, :, 1]

    # Define the new grid in the projected space
    x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
    y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)
    x_new = np.arange(x_min, x_max, cell_size_m)
    y_new = np.arange(y_min, y_max, cell_size_m)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

    # Prepare arguments for multiprocessing
    args = [
        (ds[varname].isel(time=t).values, x_coords, y_coords, x_new_grid, y_new_grid)
        for t in range(ds.dims['time'])
    ]

    # Use multiprocessing to process all time steps
    with Pool(processes=N_PROCESSORS) as pool:
        resampled_values = pool.map(reproject_and_resample_time_step, args)

    # Stack the resampled values along the time dimension
    resampled_values = np.stack(resampled_values, axis=0)

    # Create a new xarray dataset with the resampled data
    new_ds = xr.Dataset(
        {
            varname: (['time', 'y', 'x'], resampled_values)
        },
        coords={
            'time': ds['time'].values,
            'x': x_new,
            'y': y_new
        }
    )

    return new_ds, x_new, y_new

def resample_to_latlon_time_step(args):
    """Function to resample from (x, y) back to the original lat/lon grid for a single time step."""
    values_resampled, x_new_grid, y_new_grid, lon_grid, lat_grid = args

    # Flatten the resampled grid for interpolation
    x_flat = x_new_grid.flatten()
    y_flat = y_new_grid.flatten()
    var_flat = values_resampled.flatten()

    # Interpolate back to the original lat/lon grid
    var_original = griddata(
        points=(x_flat, y_flat),
        values=var_flat,
        xi=(lon_grid, lat_grid),
        method='linear'
    )
    return var_original


def reproject_raster_back2latlon(ds_proj, ds_original, varname, target_projection=ccrs.Sinusoidal(), original_projection=ccrs.PlateCarree()):
    # Extract original latitude and longitude
    latitudes = ds_original.coords['lat'].values
    longitudes = ds_original.coords['lon'].values

    # Create a mesh grid of original lat/lon
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Transform original lat/lon to x/y in the target projection
    points = target_projection.transform_points(original_projection, lon_grid, lat_grid)
    x_coords = points[:, :, 0]
    y_coords = points[:, :, 1]

    # Extract resampled x/y grids
    x_new = ds_proj['x'].values
    y_new = ds_proj['y'].values
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

    # Prepare arguments for multiprocessing
    args = [
        (ds_proj[varname].isel(time=t).values, x_new_grid, y_new_grid, x_coords, y_coords)
        for t in range(ds_proj.dims['time'])
    ]

    # Use multiprocessing to process all time steps
    with Pool(processes=N_PROCESSORS) as pool:
        original_values = pool.map(resample_to_latlon_time_step, args)

    # Stack the resampled values back to the original lat/lon grid along the time dimension
    original_values = np.stack(original_values, axis=0)

    # Create a new xarray dataset with the original resolution and projection
    new_ds = xr.Dataset(
        {
            varname: (['time', 'lat', 'lon'], original_values)
        },
        coords={
            'time': ds_proj['time'].values,
            'lat': latitudes,
            'lon': longitudes
        }
    )

    return new_ds