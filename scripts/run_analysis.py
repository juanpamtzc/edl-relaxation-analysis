import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse

# Get the absolute path of the directory containing the script (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (the parent of scripts/)
project_root = os.path.join(script_dir, '..')
# Add the project root to sys.path so Python can find the 'analysis' package
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.data_processing_tools import process_filename
from analysis.angular_analysis_tools import COM_trj, compute_local_basis_unit_vectors, arrange_trj_data_by_molecules, unwrap_trj
from analysis.transient_analysis_tools import compute_region_density_over_time, compute_velocity_distribution_in_region, find_lower_wall_z_coordinates, compute_average_property_over_time
from analysis.plotting_tools import plot_time_series, plot_distribution_function, plot_2d_heatmap

# load configuration from YAML file
parser = argparse.ArgumentParser(description="Run analysis with a specified YAML configuration file.")
parser.add_argument("config", type=str, help="Path to the YAML configuration file")
args = parser.parse_args()
config_path = os.path.normpath(os.path.join(project_root, args.config))
with open(config_path) as f:
    config = yaml.safe_load(f)

#config_path = os.path.normpath(os.path.join(project_root, "configs", "20_example.yaml"))
#with open(config_path) as f:
#    config = yaml.safe_load(f)

base_path = config['base_path']
runs = config['runs']
files = config['files']
atom_types = config['atom_types']
dt = config['dt']
thermo_process_ID = config['thermo_process_ID']
trajectory_type = config['trajectory_type']

region = config['region']
density_config = config['density']
velocity_distribution_config = config['velocity_distribution']

output_config = config['output']

density_plotting_options = config['plotting_options']

# create output base directory if it doesn't exist
out_folder = os.path.join(project_root, output_config['base_folder'])
os.makedirs(out_folder, exist_ok=True)

# Global arrays for averaging the results from individual runs
global_com_density = []
global_potassium_density = []
global_chloride_density = []
global_ionic_charge_density = []
global_oxygen_xx_stress = []
global_oxygen_yy_stress = []
global_oxygen_zz_stress = []
global_interfacial_temperature = []

# loop over all the runs
for run in runs:
    print("Processing run:\t"+run)

    # set up file paths from configuration
    run_path = os.path.join(project_root, config['base_path']+ run)

    trj_file = os.path.join(run_path, files['trajectory'].format(run=run))
    vel_file = os.path.join(run_path, files['velocity'].format(run=run))
    force_file = os.path.join(run_path, files['force'].format(run=run))
    stress_file = os.path.join(run_path, files['stress'].format(run=run))

    dat_file = os.path.join(project_root, files["data"])
    thermo_file = os.path.join(project_root, "data", run + "_txt_files", files["thermo"].format(run=run))

    out_folder = os.path.join(project_root, output_config['base_folder'])
    output_prefix = os.path.join(out_folder, output_config["prefix"] + "_" + run)

    # read and process files
    trj, global2local, local2global, vel, force, stress, data, thermo_data = process_filename(trj_file,vel_file,force_file,stress_file,dat_file,thermo_file, thermo_process_ID=thermo_process_ID)

    print("thermo_data headers:")
    print(thermo_data.columns)

    # get the z-position of the carbon wall as a function of time
    lower_wall_z_coordinates = find_lower_wall_z_coordinates(trj, atom_types["carbon"], config["carbon_positions"]["zlo"])

    # extract box size from data 
    box_size = [float(data["xhi"]) - float(data["xlo"]),
                float(data["yhi"]) - float(data["ylo"]),
                float(data["zhi"]) - float(data["zlo"])]

    # get basis vectors, positions of the oxygens, h1s and h2s
    a,b,c,positions_oxygens,positions_h1s,positions_h2s = compute_local_basis_unit_vectors(data, trj, atom_types["oxygen"], atom_types["hydrogen"], box_size, global2local=global2local, mode="debug")

    # arrange velocities and forces by water molecules
    velocities_oxygens, velocities_h1s, velocities_h2s = arrange_trj_data_by_molecules(data, vel, atom_types["oxygen"], atom_types["hydrogen"], global2local=global2local)
    forces_oxygens, forces_h1s, forces_h2s = arrange_trj_data_by_molecules(data, force, atom_types["oxygen"], atom_types["hydrogen"], global2local=global2local)

    # extract the normal per-atom stresses
    per_atom_stresses = stress[:,:,:4]

    # arrange normal stresses by water molecules
    stresses_oxygens, stresses_h1s, stresses_h2s = arrange_trj_data_by_molecules(data, per_atom_stresses, atom_types["oxygen"], atom_types["hydrogen"], global2local=global2local)

    # unwrap water molecule positions
    positions_oxygens, positions_h1s, positions_h2s = unwrap_trj(positions_oxygens, positions_h1s, positions_h2s, data)

    # get positions, velocities, and forces of potassium ions
    positions_K = trj[:, (trj[0,:,0]==atom_types["potassium"]), 1:]
    velocities_K = vel[:, (vel[0,:,0]==atom_types["potassium"]), 1:]
    forces_K = force[:, (force[0,:,0]==atom_types["potassium"]), 1:]

    # get positions, velocities, and forces of chloride ions
    positions_Cl = trj[:, (trj[0,:,0]==atom_types["chloride"]), 1:]
    velocities_Cl = vel[:, (vel[0,:,0]==atom_types["chloride"]), 1:]
    forces_Cl = force[:, (force[0,:,0]==atom_types["chloride"]), 1:]

    # Compute center-of-mass positions and velocities
    positions_COM, velocities_COM = COM_trj(positions_oxygens, positions_h1s, positions_h2s,velocities_oxygens, velocities_h1s, velocities_h2s,data, atom_types["hydrogen"], atom_types["oxygen"])

    # Density analysis
    time, com_density = compute_region_density_over_time(positions_COM, zlo=region["zlo"], zhi=region["zhi"], dt=dt, cross_sectional_area=box_size[0]*box_size[1],plot=False)
    time, potassium_density = compute_region_density_over_time(positions_K, zlo=region["zlo"], zhi=region["zhi"], dt=dt, cross_sectional_area=box_size[0]*box_size[1],plot=False)
    time, chloride_density = compute_region_density_over_time(positions_Cl, zlo=region["zlo"], zhi=region["zhi"], dt=dt, cross_sectional_area=box_size[0]*box_size[1],plot=False)
    ionic_charge_density = potassium_density * 1.00000 + chloride_density * -1.00000
    
    # collect densities for global averaging
    global_com_density.append(com_density)
    global_potassium_density.append(potassium_density)
    global_chloride_density.append(chloride_density)
    global_ionic_charge_density.append(ionic_charge_density)

    # Stress analysis
    time, oxygen_xx_stress = compute_average_property_over_time(positions_COM, stresses_oxygens, zlo=region["zlo"], zhi=region["zhi"], dt=dt,component=0,plot=False)
    time, oxygen_yy_stress = compute_average_property_over_time(positions_COM, stresses_oxygens, zlo=region["zlo"], zhi=region["zhi"], dt=dt,component=1,plot=False)
    time, oxygen_zz_stress = compute_average_property_over_time(positions_COM, stresses_oxygens, zlo=region["zlo"], zhi=region["zhi"], dt=dt,component=2,plot=False)

    # collect stresses for global averaging
    global_oxygen_xx_stress.append(oxygen_xx_stress)
    global_oxygen_yy_stress.append(oxygen_yy_stress)
    global_oxygen_zz_stress.append(oxygen_zz_stress)

    # collect interfacial temperatures for global averaging
    global_interfacial_temperature.append(thermo_data['c_T1_xy'])
    thermo_time = thermo_data['Step']

# average densities over all runs
avg_com_density = np.mean(np.array(global_com_density), axis=0)
avg_potassium_density = np.mean(np.array(global_potassium_density), axis=0)
avg_chloride_density = np.mean(np.array(global_chloride_density), axis=0)
avg_ionic_charge_density = np.mean(np.array(global_ionic_charge_density), axis=0)
# standard deviation for error bars
std_com_density = np.std(np.array(global_com_density), axis=0)
std_potassium_density = np.std(np.array(global_potassium_density), axis=0)
std_chloride_density = np.std(np.array(global_chloride_density), axis=0)
std_ionic_charge_density = np.std(np.array(global_ionic_charge_density), axis=0)

# average stresses over all runs
avg_oxygen_xx_stress = np.mean(np.array(global_oxygen_xx_stress), axis=0)
avg_oxygen_yy_stress = np.mean(np.array(global_oxygen_yy_stress), axis=0)
avg_oxygen_zz_stress = np.mean(np.array(global_oxygen_zz_stress), axis=0)
# standard deviation for error bars
std_oxygen_xx_stress = np.std(np.array(global_oxygen_xx_stress), axis=0)
std_oxygen_yy_stress = np.std(np.array(global_oxygen_yy_stress), axis=0)
std_oxygen_zz_stress = np.std(np.array(global_oxygen_zz_stress), axis=0)

# average interfacial temperature over all runs
avg_interfacial_temperature = np.mean(np.array(global_interfacial_temperature), axis=0)
std_interfacial_temperature = np.std(np.array(global_interfacial_temperature), axis=0)

# set timeseries xlim
if "time_series_max_limit" not in density_plotting_options:
    xlim_timeseries = (0, density_plotting_options["time_series_max_limit"])

# plot averaged densities
plot_time_series(time, avg_com_density, title="Average COM Density Over Time", xlabel="Time (fs)", ylabel="Density (molecules/Å^3)", output_file=output_prefix+"_avg_com_density.png", xlim=xlim_timeseries)
plot_time_series(time, avg_potassium_density, title="Average Potassium Density Over Time", xlabel="Time (fs)", ylabel="Density (ions/Å^3)", output_file=output_prefix+"_avg_potassium_density.png", xlim=xlim_timeseries)
plot_time_series(time, avg_chloride_density, title="Average Chloride Density Over Time", xlabel="Time (fs)", ylabel="Density (ions/Å^3)", output_file=output_prefix+"_avg_chloride_density.png", xlim=xlim_timeseries)
plot_time_series(time, avg_ionic_charge_density, title="Average Ionic Charge Density Over Time", xlabel="Time (fs)", ylabel="Charge Density (e/Å^3)", output_file=output_prefix+"_avg_ionic_charge_density.png", xlim=xlim_timeseries)
plot_time_series(time, avg_oxygen_xx_stress, title="Average Oxygen XX Stress Over Time", xlabel="Time (fs)", ylabel="Stress", output_file=output_prefix+"_avg_oxygen_xx_stress.png", xlim=xlim_timeseries)
plot_time_series(time, avg_oxygen_yy_stress, title="Average Oxygen YY Stress Over Time", xlabel="Time (fs)", ylabel="Stress", output_file=output_prefix+"_avg_oxygen_yy_stress.png", xlim=xlim_timeseries)
plot_time_series(time, avg_oxygen_zz_stress, title="Average Oxygen ZZ Stress Over Time", xlabel="Time (fs)", ylabel="Stress", output_file=output_prefix+"_avg_oxygen_zz_stress.png", xlim=xlim_timeseries)
plot_time_series(thermo_time, avg_interfacial_temperature, title="Average Interfacial Temperature Over Time", xlabel="Time (fs)", ylabel="Temperature (K)", output_file=output_prefix+"_avg_interfacial_temperature.png", xlim=xlim_timeseries)  

# plot hysteresis loops
if trajectory_type!="process":
    plot_time_series(lower_wall_z_coordinates, avg_com_density, title="Hysteresis Loop: COM Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="COM Density (molecules/Å^3)", output_file=output_prefix+"_hysteresis_com_density.png", xlim=(density_plotting_options['water_density_xlo'], density_plotting_options['water_density_xhi']), ylim=(density_plotting_options['water_density_ylo'], density_plotting_options['water_density_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_potassium_density, title="Hysteresis Loop: Potassium Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Potassium Density (ions/Å^3)", output_file=output_prefix+"_hysteresis_potassium_density.png", xlim=(density_plotting_options['potassium_density_xlo'], density_plotting_options['potassium_density_xhi']), ylim=(density_plotting_options['potassium_density_ylo'], density_plotting_options['potassium_density_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_chloride_density, title="Hysteresis Loop: Chloride Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Chloride Density (ions/Å^3)", output_file=output_prefix+"_hysteresis_chloride_density.png", xlim=(density_plotting_options['chloride_density_xlo'], density_plotting_options['chloride_density_xhi']), ylim=(density_plotting_options['chloride_density_ylo'], density_plotting_options['chloride_density_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_ionic_charge_density, title="Hysteresis Loop: Ionic Charge Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Ionic Charge Density (e/Å^3)", output_file=output_prefix+"_hysteresis_ionic_charge_density.png", xlim=(density_plotting_options['ionic_charge_density_xlo'], density_plotting_options['ionic_charge_density_xhi']), ylim=(density_plotting_options['ionic_charge_density_ylo'], density_plotting_options['ionic_charge_density_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_oxygen_xx_stress, title="Hysteresis Loop: Oxygen XX Stress vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Oxygen XX Stress", output_file=output_prefix+"_hysteresis_oxygen_xx_stress.png", xlim=(density_plotting_options['xx_stress_xlo'], density_plotting_options['xx_stress_xhi']), ylim=(density_plotting_options['xx_stress_ylo'], density_plotting_options['xx_stress_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_oxygen_yy_stress, title="Hysteresis Loop: Oxygen YY Stress vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Oxygen YY Stress", output_file=output_prefix+"_hysteresis_oxygen_yy_stress.png", xlim=(density_plotting_options['yy_stress_xlo'], density_plotting_options['yy_stress_xhi']), ylim=(density_plotting_options['yy_stress_ylo'], density_plotting_options['yy_stress_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_oxygen_zz_stress, title="Hysteresis Loop: Oxygen ZZ Stress vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Oxygen ZZ Stress", output_file=output_prefix+"_hysteresis_oxygen_zz_stress.png", xlim=(density_plotting_options['zz_stress_xlo'], density_plotting_options['zz_stress_xhi']), ylim=(density_plotting_options['zz_stress_ylo'], density_plotting_options['zz_stress_yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_interfacial_temperature[:-1], title="Hysteresis Loop: Interfacial Temperature vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Interfacial Temperature (K)", output_file=output_prefix+"_hysteresis_interfacial_temperature.png", xlim=(density_plotting_options['temp_xlo'], density_plotting_options['temp_xhi']), ylim=(density_plotting_options['temp_ylo'], density_plotting_options['temp_yhi']))