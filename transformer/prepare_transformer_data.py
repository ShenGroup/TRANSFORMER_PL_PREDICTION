import rasterio
import numpy as np
import csv
import os
import pandas as pd
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from math import sin, cos, atan2, sqrt, radians, degrees
import matplotlib.pyplot as plt
import re

def great_circle_interpolation(lat1, lon1, lat2, lon2, num_samples):
    """
    Computes points along a great-circle path between two lat/lon coordinates.
    
    Args:
        lat1 (float): Latitude of Tx (start point) in degrees.
        lon1 (float): Longitude of Tx (start point) in degrees.
        lat2 (float): Latitude of Rx (end point) in degrees.
        lon2 (float): Longitude of Rx (end point) in degrees.
        num_samples (int): Number of points to sample along the path (including endpoints).
    
    Returns:
        list: A list of (lat, lon) tuples along the great-circle path.
    """
    # Convert to radians
    phi1 = radians(lat1)
    lambda1 = radians(lon1)
    phi2 = radians(lat2)
    lambda2 = radians(lon2)
    
    # Compute central angle d using haversine formula
    d_lambda = lambda2 - lambda1
    cos_d = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)
    # Clamp to avoid numerical issues
    cos_d = max(-1.0, min(1.0, cos_d))
    d = atan2(sqrt(1 - cos_d**2), cos_d)
    
    # If points are the same or very close, return just the start point
    if d < 1e-9:
        return [(lat1, lon1)]
    
    points = []
    for i in range(num_samples):
        f = i / (num_samples - 1) if num_samples > 1 else 0.0
        
        # Spherical interpolation coefficients
        A = sin((1 - f) * d) / sin(d)
        B = sin(f * d) / sin(d)
        
        # Convert to ECEF coordinates
        x = A * cos(phi1) * cos(lambda1) + B * cos(phi2) * cos(lambda2)
        y = A * cos(phi1) * sin(lambda1) + B * cos(phi2) * sin(lambda2)
        z = A * sin(phi1) + B * sin(phi2)
        
        # Convert back to geodetic coordinates
        phi_i = atan2(z, sqrt(x*x + y*y))
        lambda_i = atan2(y, x)
        
        # Convert back to degrees
        lat_i = degrees(phi_i)
        lon_i = degrees(lambda_i)
        
        points.append((lat_i, lon_i))
    
    return points

def compute_los_nlos(heights, distances, tx_height, rx_height, effective_earth_radius_factor=4/3):
    """
    Computes Line of Sight (LOS) or Non-Line of Sight (NLOS) for a path.
    
    Args:
        heights (list or np.array): Terrain heights along the path in meters.
        distances (list or np.array): Distances from Tx along the path in meters.
        tx_height (float): Transmitter antenna height above terrain in meters.
        rx_height (float): Receiver antenna height above terrain in meters.
        effective_earth_radius_factor (float): Effective Earth radius factor (default 4/3 for radio).
    
    Returns:
        tuple: (is_los (bool), clearance_profile (np.array))
    """
    heights = np.array(heights)
    distances = np.array(distances)
    
    if len(heights) < 2 or len(distances) < 2:
        return False, np.array([])
    
    # Earth radius in meters
    R_earth = 6371000.0
    R_eff = R_earth * effective_earth_radius_factor
    
    # Total path distance
    total_distance = distances[-1]
    
    # Heights above sea level (terrain + antenna)
    tx_height_above_sea = heights[0] + tx_height
    rx_height_above_sea = heights[-1] + rx_height
    
    # Compute clearance profile: height of line of sight above terrain
    # Accounting for Earth's curvature
    clearance = np.zeros_like(heights)
    
    for i in range(len(heights)):
        d = distances[i]  # Distance from Tx
        # Height of LOS line at this point (linear interpolation + Earth curvature correction)
        # Earth curvature correction: h_curve = d * (total_distance - d) / (2 * R_eff)
        h_curve = d * (total_distance - d) / (2 * R_eff)
        los_height = tx_height_above_sea + (rx_height_above_sea - tx_height_above_sea) * (d / total_distance) - h_curve
        clearance[i] = los_height - heights[i]
    
    # LOS if all points have positive clearance (with small margin for numerical errors)
    is_los = np.all(clearance > -0.1)  # Small margin to account for numerical precision
    
    return is_los, clearance

def plot_path_profile(heights, pathlosslabels, distances, is_los, clearance, 
                     output_path=None, tx_height=0, rx_height=0):
    """
    Plots three subfigures: power profile, height profile, and LOS/NLOS visualization.
    
    Args:
        heights (list or np.array): Terrain heights along the path.
        pathlosslabels (list or np.array): Path loss values along the path.
        distances (list or np.array): Distances from Tx along the path.
        is_los (bool): Whether the path is LOS or NLOS.
        clearance (np.array): Clearance profile (LOS height - terrain height).
        output_path (str): Path to save the plot. If None, displays the plot.
        tx_height (float): Transmitter antenna height.
        rx_height (float): Receiver antenna height.
    """
    heights = np.array(heights)
    pathlosslabels = np.array(pathlosslabels)
    distances = np.array(distances)
    clearance = np.array(clearance)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Subfigure 1: Power profile (Path Loss)
    ax1 = axes[0]
    ax1.plot(distances, pathlosslabels, 'b-', linewidth=2, label='Path Loss')
    ax1.set_xlabel('Distance from Tx (m)', fontsize=12)
    ax1.set_ylabel('Path Loss (dB)', fontsize=12)
    ax1.set_title(f'Power Profile (Path Loss) - {"LOS" if is_los else "NLOS"}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Subfigure 2: Height profile
    ax2 = axes[1]
    ax2.plot(distances, heights, 'g-', linewidth=2, label='Terrain Height')
    if tx_height > 0 or rx_height > 0:
        tx_height_above_sea = heights[0] + tx_height
        rx_height_above_sea = heights[-1] + rx_height
        # Plot antenna heights
        ax2.plot(distances[0], tx_height_above_sea, 'ro', markersize=10, label=f'Tx (h={tx_height}m)')
        ax2.plot(distances[-1], rx_height_above_sea, 'rs', markersize=10, label=f'Rx (h={rx_height}m)')
    ax2.set_xlabel('Distance from Tx (m)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Height Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subfigure 3: LOS/NLOS visualization (Clearance profile)
    ax3 = axes[2]
    colors = ['green' if c > 0 else 'red' for c in clearance]
    ax3.fill_between(distances, 0, clearance, color='green', alpha=0.3, label='Clear LOS')
    ax3.fill_between(distances, clearance, 0, where=(clearance < 0), color='red', alpha=0.3, label='Blocked')
    ax3.plot(distances, clearance, 'k-', linewidth=2, label='Clearance Profile')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Distance from Tx (m)', fontsize=12)
    ax3.set_ylabel('Clearance (m)', fontsize=12)
    los_status = "LOS" if is_los else "NLOS"
    ax3.set_title(f'LOS/NLOS Analysis - {los_status}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {output_path}')
    else:
        plt.show()
    
    plt.close()

def get_dots_on_line(x1, y1, x2, y2, distance_interval):
    """
    Generates a list of (x, y) coordinates for dots placed every
    distance_interval along the line from (x1, y1) to (x2, y2).
    Includes the start and end points.

    Args:
        x1 (float): x-coordinate of the start point.
        y1 (float): y-coordinate of the start point.
        x2 (float): x-coordinate of the end point.
        y2 (float): y-coordinate of the end point.
        distance_interval (float): The fixed distance between dots.

    Returns:
        list: A list of (x, y) tuples.
    """
    if distance_interval <= 0:
        raise ValueError("distance_interval must be positive.")

    dots = []
    start_point = np.array([x1, y1])
    end_point = np.array([x2, y2])

    # Add the starting point
    dots.append((x1, y1))

    # Calculate the vector from start to end and its length
    vector = end_point - start_point
    line_length = np.linalg.norm(vector)

    # If the line length is 0 (start and end points are the same),
    # we've already added the start point, so we're done.
    if line_length == 0:
        return dots

    # Calculate the unit vector (direction of the line)
    unit_vector = vector / line_length

    # Add intermediate dots
    accumulated_distance = distance_interval
    while accumulated_distance < line_length:
        # Calculate the next point along the line
        current_point = start_point + unit_vector * accumulated_distance
        dots.append((current_point[0], current_point[1]))
        accumulated_distance += distance_interval

    return dots

def get_profile_heights(dsm_path, output_filename, frq, height, pwr, rad, pol, tx_lat=None, tx_lon=None, rx_lat=None, rx_lon=None, step_size_meters=None, create_plots=False, plot_every_nth=100):
    """
    Calculates elevation profiles along a great-circle path on a DSM.

    Args:
        dsm_path (str): Path to the DSM file (e.g., a GeoTIFF).
        output_filename (str): Path to output CSV file.
        frq, height, pwr, rad, pol: Radio parameters.
        tx_lat (float): Latitude of Tx (transmitter) in degrees. If None, uses center of DSM.
        tx_lon (float): Longitude of Tx (transmitter) in degrees. If None, uses center of DSM.
        rx_lat (float): Latitude of Rx (receiver) in degrees. If None, computes from DSM bounds.
        rx_lon (float): Longitude of Rx (receiver) in degrees. If None, computes from DSM bounds.
        step_size_meters (float): Step size along the great-circle path in meters. 
                                  If None, uses ~3-5x smaller than DEM spacing.
        create_plots (bool): Whether to create plots. Default True.
        plot_every_nth (int): Only plot every Nth path to avoid too many plots. Default 10.
    """

    # Import transform at function level
    from rasterio.warp import transform
    
    # Open the DSM and sample the raster values at the generated points
    with rasterio.open(dsm_path) as src:
        with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
            # The 'sample' method takes an iterable of (x, y) coordinates
            # and returns a generator of the band values at those locations.
            detect_radiusx = src.res[0]
            detect_radiusy = src.res[1]
            assert(detect_radiusx==detect_radiusy)
            
            # Get center point in projected coordinates
            center_x = (src.bounds[0] + src.bounds[2]) / 2
            center_y = (src.bounds[1] + src.bounds[3]) / 2
            
            # Get center point in lat/lon - this is Tx
            if tx_lat is None or tx_lon is None:
                # Convert center point from projected to lat/lon
                tx_lon, tx_lat = transform(src.crs, 'EPSG:4326', [center_x], [center_y])
                tx_lat, tx_lon = tx_lat[0], tx_lon[0]
            
            # Sample center point for input values
            sampled_generator = vrt.sample([(center_x, center_y)])
            value_in = next(sampled_generator)
            pathloss_in = -value_in[0]
            height_in = value_in[1]
            
            # Get DSM dimensions
            ylen, xlen = src.shape
            
            # Calculate crop bounds for center 256x256
            crop_size = 256
            start_x = xlen // 2 - crop_size // 2
            start_y = ylen // 2 - crop_size // 2
            
            # Ensure valid bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(xlen, start_x + crop_size)
            end_y = min(ylen, start_y + crop_size)

            # Calculate step size: use ~3-5x smaller than DEM spacing, or ~90-100m
            if step_size_meters is None:
                # Use approximately 3-5x smaller than DEM spacing
                step_size_meters = 100

            # Iterate through grid points (i, j) - each becomes an Rx point
            # i is row index (y direction), j is column index (x direction)
            for i in range(start_y, end_y):
                for j in range(start_x, end_x):
                    # Convert (i, j) grid indices to projected coordinates (x, y)
                    # Using transform matrix: x = transform[2] + j*transform[0] + i*transform[1]
                    #                          y = transform[5] + j*transform[3] + i*transform[4]
                    x = src.transform[2] + j * src.transform[0] + i * src.transform[1]
                    y = src.transform[5] + j * src.transform[3] + i * src.transform[4]
                    
                    # Convert projected coordinates to lat/lon - this is Rx
                    rx_lon, rx_lat = transform(src.crs, 'EPSG:4326', [x], [y])
                    rx_lat, rx_lon = rx_lat[0], rx_lon[0]
                    
                    # Calculate great-circle distance between Tx and Rx
                    phi1 = radians(tx_lat)
                    lambda1 = radians(tx_lon)
                    phi2 = radians(rx_lat)
                    lambda2 = radians(rx_lon)
                    d_lambda = lambda2 - lambda1
                    cos_d = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)
                    cos_d = max(-1.0, min(1.0, cos_d))
                    d = atan2(sqrt(1 - cos_d**2), cos_d)
                    
                    # Earth radius in meters
                    R_earth = 6371000.0
                    total_distance = d * R_earth
                    
                    # Skip if Tx and Rx are too close
                    if total_distance < 1.0:  # Less than 1 meter
                        continue
                    if total_distance > rad:
                        continue
                    
                    # Number of samples along the path
                    num_samples = max(2, int(total_distance / step_size_meters) + 1)
                    
                    # Get great-circle path points from Tx to Rx
                    gc_points = great_circle_interpolation(tx_lat, tx_lon, rx_lat, rx_lon, num_samples)
                    
                    # Convert great-circle points to projected coordinates for sampling
                    dots = []
                    dot_x_test, dot_y_test = transform('EPSG:4326', src.crs, [rx_lon], [rx_lat])
                    dots_test = [(dot_x_test[0], dot_y_test[0])]
                    sampled_heights_generator_test = vrt.sample(dots_test)
                    values_test = [value_test for value_test in sampled_heights_generator_test]
                    pathlosslabels_test = [-float(value_test[0]) for value_test in values_test]
                    if abs(pathlosslabels_test[-1]) > 1e37:
                        continue

                    for lat_dot, lon_dot in gc_points:
                        # Convert to projected coordinates
                        dot_x, dot_y = transform('EPSG:4326', src.crs, [lon_dot], [lat_dot])
                        dots.append((dot_x[0], dot_y[0]))
                    
                    sampled_heights_generator = vrt.sample(dots)
                    
                    values = [value for value in sampled_heights_generator]
                    heights = [float(value[1]) for value in values]
                    pathlosslabels = [-float(value[0]) for value in values]

                    
                    # Compute distances along the path (cumulative great-circle distances)
                    distances = []
                    R_earth = 6371000.0
                    cumulative_dist = 0.0
                    distances.append(0.0)  # Start at Tx
                    
                    for k in range(1, len(gc_points)):
                        # Compute great-circle distance between consecutive points
                        lat1, lon1 = gc_points[k-1]
                        lat2, lon2 = gc_points[k]
                        phi1 = radians(lat1)
                        lambda1 = radians(lon1)
                        phi2 = radians(lat2)
                        lambda2 = radians(lon2)
                        d_lambda = lambda2 - lambda1
                        cos_d = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)
                        cos_d = max(-1.0, min(1.0, cos_d))
                        d = atan2(sqrt(1 - cos_d**2), cos_d)
                        segment_dist = d * R_earth
                        cumulative_dist += segment_dist
                        distances.append(cumulative_dist)
                    
                    distances = np.array(distances)
                    
                    # Compute distances from Rx (end point) for each sample point
                    total_distance = distances[-1]
                    distances_from_rx = total_distance - distances
                    
                    # Compute LOS/NLOS
                    tx_height_above_terrain = height  # Antenna height parameter
                    rx_height_above_terrain = 50  # Receiver height (from CSV)
                    is_los, clearance = compute_los_nlos(heights, distances, tx_height_above_terrain, rx_height_above_terrain)
                    
                    print(f'Grid point (i={i}, j={j}): pathlosslabels[-1] = {pathlosslabels[-1]}')
                    print(f'pathlosslabels[0] = {pathlosslabels[0]}')
                    print(f'pathlosslabels[1] = {pathlosslabels[1]}')
                    print(f'heights[-1] = {heights[-1]}')
                    if heights[-1] < 0:
                        continue
                    print(f'heights[0] = {heights[0]}')
                    print(f'LOS Status: {"LOS" if is_los else "NLOS"}')
                    
                    # Create plot for this path (if enabled and meets criteria)
                    if create_plots and (i * xlen + j) % plot_every_nth == 0:

                        plot_dir = os.path.join(os.path.dirname(output_filename), os.path.basename(output_filename) +'plots')
                        os.makedirs(plot_dir, exist_ok=True)
                        plot_filename = os.path.join(plot_dir, f'path_i{i}_j{j}_los{int(is_los)}.png')
                        plot_path_profile(heights, pathlosslabels, distances, is_los, clearance,
                                         output_path=plot_filename, tx_height=tx_height_above_terrain, 
                                         rx_height=rx_height_above_terrain)
                    
                    outputfile_num = re.sub(r'(\.csv)$', f'_{int(is_los)}\\1', output_filename)
                    file_exists = os.path.isfile(outputfile_num)
                    with open(outputfile_num, 'a', newline='') as f:
                        # Create a CSV writer object
                        writer = csv.writer(f)
                        # Write the header row if the file is new
                        if not file_exists:
                            writer.writerow(["path_loss_in", "height_in", "height_profile", "pathlosslabel_profile", 
                                            "distance_from_tx_profile", "distance_from_rx_profile",
                                            "detect_res", "i", "j", "tx_lat", "tx_lon", "rx_lat", "rx_lon",
                                            "frq", "height", "pwr", "pol", "rad", "receiver_height", "is_los"])

                        # Convert the list of heights into a single string
                        height_str = ";".join(map(str, heights))
                        pathlosslabel_str = ";".join(map(str, pathlosslabels))
                        distance_from_tx_str = ";".join(map(str, distances))
                        distance_from_rx_str = ";".join(map(str, distances_from_rx))
                        
                        # Write the new row to the CSV file
                        writer.writerow([pathloss_in, height_in, height_str, pathlosslabel_str, 
                                        distance_from_tx_str, distance_from_rx_str,
                                        detect_radiusx, i, j, tx_lat, tx_lon, rx_lat, rx_lon,
                                        frq, height, pwr, pol, rad, 50, int(is_los)])
# --- --- --- --- --- --- ---
# EXAMPLE USAGE
# --- --- --- --- --- --- ---

# 1. Define your inputs 
df_params = pd.read_csv("/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/parameters.csv")
df_file = pd.read_csv("/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/analysis_catalog.csv")
counter = 0
for pid,filename,oid in zip(df_file['PID'].values,df_file['RID'].values,df_file['OID'].values):
    counter += 1
    print(counter)
    dsmfile = '/scratch/tvs9by/GPT2/trainingdata_new/datat/datat/' + filename + '.tiff'
    params = df_params[df_params['PID']==pid].values[0]
    frq,pol,height,pwr,rad = params[1],params[2],params[3],params[4],params[5]
    if pol == 'Horizontal':
        pol_in = 0
    else:
        pol_in = 1
    if 121 <= oid and oid <= 121 :
        if oid <= 181:
            outputfile = '/scratch/tvs9by/ntia/pathprofile/collect_data/data_collected/training/' + f'training_v4_bilinear_oid_{oid}.csv'
            get_profile_heights(dsmfile,outputfile,frq,height,pwr,rad,pol_in)
        else:
            outputfile = '/scratch/tvs9by/ntia/pathprofile/collect_data/data_collected/testing/' + f'testing_v4_bilinear_oid_{oid}.csv'
            get_profile_heights(dsmfile,outputfile,frq,height,pwr,rad,pol_in)
# dsm_file = '/home/lican/workarea/Zihaoliang/machine_test/datat2/TIREM_0_031925_070015_7723.tiff' # IMPORTANT: Replace with the actual path to your DSM
# get_profile_heights(dsm_file)
