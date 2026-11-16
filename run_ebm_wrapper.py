import subprocess
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

def compile_and_run_ebm():
    # Directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'EBM_Zuang_North_Stevens'))
    src_dir = os.path.join(base_dir, 'src')
    output_dir = os.path.join(base_dir, 'output')
    
    print("Compiling Fortran EBM...")
    compile_cmd = "make"
    subprocess.run(compile_cmd, cwd=src_dir, shell=True, check=True)
    
    print("Running Fortran EBM...")
    run_cmd = "./EBM"
    subprocess.run(run_cmd, cwd=src_dir, shell=True, check=True)
    
    print("Run completed. Reading outputs...")
    
    # Read the final timestep NetCDF output
    # EBM produces 'timesteps-output.nc' and 'monthly-output.nc' in the output directory
    nc_file = os.path.join(output_dir, 'timesteps-output.nc')
    
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"Output file {nc_file} not found. The model might not have run correctly.")
        
    ds = nc.Dataset(nc_file, 'r')
    
    # Read variables
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    temp = ds.variables['temperature'][:]
    
    # Calculate global mean for the final year (last 48 timesteps)
    # The output is [time, lat, lon]
    # The output is [time, level, lat, lon] -> squeeze out level dimension
    temp_last_year = temp[-48:, 0, :, :]
    mean_temp_2d = np.mean(temp_last_year, axis=0)
    
    # Print the result
    print(f"Data saved successfully. Final shape: {temp.shape}")
    print(f"Mean Global Temperature in final year (approximate): {np.mean(mean_temp_2d):.4f} C")
    
    # Save a plot for inspection
    plt.figure(figsize=(10, 5))
    plt.contourf(lon, lat, mean_temp_2d, levels=20, cmap='inferno')
    plt.colorbar(label='Temperature (C)')
    plt.title('EBM Final Year Average Temperature')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../figures'))
    if not os.path.exists(fig_dir): fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, 'wrapper_temperature.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
if __name__ == '__main__':
    compile_and_run_ebm()
