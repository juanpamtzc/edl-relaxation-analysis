import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Get the absolute path of the directory containing the script (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (the parent of scripts/)
project_root = os.path.join(script_dir, '..')
# Add the project root to sys.path so Python can find the 'analysis' package
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.data_processing_tools import process_filename, save_timeseries_as_txt, read_timeseries_from_txt
from analysis.angular_analysis_tools import COM_trj, compute_local_basis_unit_vectors, arrange_trj_data_by_molecules, unwrap_trj
from analysis.transient_analysis_tools import compute_region_density_over_time, compute_velocity_distribution_in_region, find_lower_wall_z_coordinates, compute_average_property_over_time, compute_transient_density_profile
from analysis.plotting_tools import plot_time_series, plot_distribution_function, plot_2d_heatmap, plot_loglog, plot_semilogy
from analysis.smoothing_tools import find_critical_points_via_spline_fitting

# load configuration from YAML file
parser = argparse.ArgumentParser(description="Run analysis with a specified YAML configuration file.")
parser.add_argument("config", type=str, help="Path to the YAML configuration file")
args = parser.parse_args()
config_path = os.path.normpath(os.path.join(project_root, args.config))
with open(config_path) as f:
    config = yaml.safe_load(f)

# create output base directory if it doesn't exist
out_folder = os.path.join(project_root, config['output']['base_folder'])
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
global_transient_com_density = []
global_transient_potassium_density = []
global_transient_chloride_density = []

# loop over all the runs
for run in config['runs']:
    print("Processing run:\t"+run)

    # set up file paths from configuration
    run_path = os.path.join(project_root, config['base_path']+ run)

    trj_file = os.path.join(run_path, config['files']['trajectory'].format(run=run))
    vel_file = os.path.join(run_path, config['files']['velocity'].format(run=run))
    force_file = os.path.join(run_path, config['files']['force'].format(run=run))
    stress_file = os.path.join(run_path, config['files']['stress'].format(run=run))

    dat_file = os.path.join(project_root, config['files']["data"])
    thermo_file = os.path.join(project_root, "data", run + "_txt_files", config['files']["thermo"].format(run=run))

    out_folder = os.path.join(project_root, config['output']['base_folder'])
    output_prefix = os.path.join(out_folder, config['output']["prefix"])

    # read and process files
    trj, global2local, local2global, vel, force, stress, data, thermo_data = process_filename(trj_file,vel_file,force_file,stress_file,dat_file,thermo_file, thermo_process_ID=config['thermo_process_ID'])

    print("thermo_data headers:")
    print(thermo_data.columns)

    # get the z-position of the carbon wall as a function of time
    lower_wall_z_coordinates = find_lower_wall_z_coordinates(trj, config['atom_types']["carbon"], config["carbon_positions"]["zlo"])

    # extract box size from data 
    box_size = [float(data["xhi"]) - float(data["xlo"]),
                float(data["yhi"]) - float(data["ylo"]),
                float(data["zhi"]) - float(data["zlo"])]

    # get basis vectors, positions of the oxygens, h1s and h2s
    a,b,c,positions_oxygens,positions_h1s,positions_h2s = compute_local_basis_unit_vectors(data, trj, config['atom_types']["oxygen"], config['atom_types']["hydrogen"], box_size, global2local=global2local, mode="debug")

    # arrange velocities and forces by water molecules
    velocities_oxygens, velocities_h1s, velocities_h2s = arrange_trj_data_by_molecules(data, vel, config['atom_types']["oxygen"], config['atom_types']["hydrogen"], global2local=global2local)
    forces_oxygens, forces_h1s, forces_h2s = arrange_trj_data_by_molecules(data, force, config['atom_types']["oxygen"], config['atom_types']["hydrogen"], global2local=global2local)
    # extract the normal per-atom stresses
    per_atom_stresses = stress[:,:,:4]

    # arrange normal stresses by water molecules
    stresses_oxygens, stresses_h1s, stresses_h2s = arrange_trj_data_by_molecules(data, per_atom_stresses, config['atom_types']["oxygen"], config['atom_types']["hydrogen"], global2local=global2local)

    # unwrap water molecule positions
    positions_oxygens, positions_h1s, positions_h2s = unwrap_trj(positions_oxygens, positions_h1s, positions_h2s, data)

    # get positions, velocities, and forces of potassium ions
    positions_K = trj[:, (trj[0,:,0]==config['atom_types']["potassium"]), 1:]
    velocities_K = vel[:, (vel[0,:,0]==config['atom_types']["potassium"]), 1:]
    forces_K = force[:, (force[0,:,0]==config['atom_types']["potassium"]), 1:]

    # get positions, velocities, and forces of chloride ions
    positions_Cl = trj[:, (trj[0,:,0]==config['atom_types']["chloride"]), 1:]
    velocities_Cl = vel[:, (vel[0,:,0]==config['atom_types']["chloride"]), 1:]
    forces_Cl = force[:, (force[0,:,0]==config['atom_types']["chloride"]), 1:]

    # Compute center-of-mass positions and velocities
    positions_COM, velocities_COM = COM_trj(positions_oxygens, positions_h1s, positions_h2s,velocities_oxygens, velocities_h1s, velocities_h2s,data, config['atom_types']["hydrogen"], config['atom_types']["oxygen"])

    # Density analysis
    time, com_density = compute_region_density_over_time(positions_COM, zlo=config['region']["zlo"], zhi=config['region']["zhi"], dt=config['dt'], cross_sectional_area=box_size[0]*box_size[1],plot=False)
    time, potassium_density = compute_region_density_over_time(positions_K, zlo=config['region']["zlo"], zhi=config['region']["zhi"], dt=config['dt'], cross_sectional_area=box_size[0]*box_size[1],plot=False)
    time, chloride_density = compute_region_density_over_time(positions_Cl, zlo=config['region']["zlo"], zhi=config['region']["zhi"], dt=config['dt'], cross_sectional_area=box_size[0]*box_size[1],plot=False)
    ionic_charge_density = potassium_density * 1.00000 + chloride_density * -1.00000

    if config['trajectory_type']=='process':
        bin_centers, density_profile_COM, time = compute_transient_density_profile(positions_COM, zlo=config['transient_density_profile']['zlo'], zhi=config['transient_density_profile']['zhi'], bin_width=config['transient_density_profile']['bin_width'], dt=config['dt'], plot=False)
        bin_centers, density_profile_K, time = compute_transient_density_profile(positions_K, zlo=config['transient_density_profile']['zlo'], zhi=config['transient_density_profile']['zhi'], bin_width=config['transient_density_profile']['bin_width'], dt=config['dt'], plot=False)
        bin_centers, density_profile_Cl, time = compute_transient_density_profile(positions_Cl, zlo=config['transient_density_profile']['zlo'], zhi=config['transient_density_profile']['zhi'], bin_width=config['transient_density_profile']['bin_width'], dt=config['dt'], plot=False)
    
    # collect densities for global averaging
    global_com_density.append(com_density)
    global_potassium_density.append(potassium_density)
    global_chloride_density.append(chloride_density)
    global_ionic_charge_density.append(ionic_charge_density)

    if config['trajectory_type']=='process':
        global_transient_com_density.append(density_profile_COM)
        global_transient_potassium_density.append(density_profile_K)
        global_transient_chloride_density.append(density_profile_Cl)

    # Stress analysis
    time, oxygen_xx_stress = compute_average_property_over_time(positions_COM, stresses_oxygens, zlo=config['region']["zlo"], zhi=config['region']["zhi"], dt=config['dt'],component=0,plot=False)
    time, oxygen_yy_stress = compute_average_property_over_time(positions_COM, stresses_oxygens, zlo=config['region']["zlo"], zhi=config['region']["zhi"], dt=config['dt'],component=1,plot=False)
    time, oxygen_zz_stress = compute_average_property_over_time(positions_COM, stresses_oxygens, zlo=config['region']["zlo"], zhi=config['region']["zhi"], dt=config['dt'],component=2,plot=False)

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

if config['trajectory_type']=='process':
    avg_com_transient_density = np.mean(np.array(global_transient_com_density), axis=0)
    avg_potassium_transient_density = np.mean(np.array(global_transient_potassium_density), axis=0)
    avg_chloride_transient_density = np.mean(np.array(global_transient_chloride_density), axis=0)

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
#if "xlo" in config['plotting_options']['time_series'] and "xhi" in config['plotting_options']['time_series']:
#    xlim_timeseries = (config['plotting_options']['time_series']["xlo"], config['plotting_options']['time_series']["xhi"])
#else:
#    xlim_timeseries = (None, None)

xlim_timeseries = (None, None)

# getting equilibrium benchmark values
equil_com_density = None
equil_potassium_density = None
equil_chloride_density = None
equil_ionic_charge_density = None
equil_oxygen_xx_stress = None
equil_oxygen_yy_stress = None
equil_oxygen_zz_stress = None
equil_interfacial_temperature = None

if 'equilibrium_benchmark_prefix' in config:
    print("Equilibrium benchmark prefix found in configuration. Loading benchmark data...")
    equilibrium_prefix = os.path.join(out_folder, config['equilibrium_benchmark_prefix'])
    if Path(equilibrium_prefix+"_avg_com_density.txt").is_file():
        print("Equilibrium COM density benchmark found. Loading equilibrium COM density benchmark data...")
        _, equil_com_density = read_timeseries_from_txt(equilibrium_prefix+"_avg_com_density.txt")
    if Path(equilibrium_prefix+"_avg_potassium_density.txt").is_file():
        print("Equilibrium potassium density benchmark found. Loading equilibrium potassium density benchmark data...")
        _, equil_potassium_density = read_timeseries_from_txt(equilibrium_prefix+"_avg_potassium_density.txt")
    if Path(equilibrium_prefix+"_avg_chloride_density.txt").is_file():
        print("Equilibrium chloride density benchmark found. Loading equilibrium chloride density benchmark data...")
        _, equil_chloride_density = read_timeseries_from_txt(equilibrium_prefix+"_avg_chloride_density.txt")
    if Path(equilibrium_prefix+"_avg_ionic_charge_density.txt").is_file():
        _, equil_ionic_charge_density = read_timeseries_from_txt(equilibrium_prefix+"_avg_ionic_charge_density.txt")
    if Path(equilibrium_prefix+"_avg_oxygen_xx_stress.txt").is_file():
        _, equil_oxygen_xx_stress = read_timeseries_from_txt(equilibrium_prefix+"_avg_oxygen_xx_stress.txt")
    if Path(equilibrium_prefix+"_avg_oxygen_yy_stress.txt").is_file():
        _, equil_oxygen_yy_stress = read_timeseries_from_txt(equilibrium_prefix+"_avg_oxygen_yy_stress.txt")
    if Path(equilibrium_prefix+"_avg_oxygen_zz_stress.txt").is_file():
        _, equil_oxygen_zz_stress = read_timeseries_from_txt(equilibrium_prefix+"_avg_oxygen_zz_stress.txt")
    if Path(equilibrium_prefix+"_avg_interfacial_temperature.txt").is_file():
        _, equil_interfacial_temperature = read_timeseries_from_txt(equilibrium_prefix+"_avg_interfacial_temperature.txt")

if config['trajectory_type']=='relaxation' and 'equilibrium_benchmark_prefix' in config:
    plot_semilogy(time, np.abs(avg_com_density - np.mean(equil_com_density)), title="Relaxation of COM Density", xlabel="Time (fs)", ylabel="|COM Density - Equilibrium COM Density| (molecules/Å^3)", output_file=output_prefix+"_relaxation_com_density.png", xlim=xlim_timeseries)
    plot_semilogy(time, np.abs(avg_potassium_density - np.mean(equil_potassium_density)), title="Relaxation of Potassium Density", xlabel="Time (fs)", ylabel="|Potassium Density - Equilibrium Potassium Density| (ions/Å^3)", output_file=output_prefix+"_relaxation_potassium_density.png", xlim=xlim_timeseries)
    plot_semilogy(time, np.abs(avg_chloride_density - np.mean(equil_chloride_density)), title="Relaxation of Chloride Density", xlabel="Time (fs)", ylabel="|Chloride Density - Equilibrium Chloride Density| (ions/Å^3)", output_file=output_prefix+"_relaxation_chloride_density.png", xlim=xlim_timeseries)
    plot_semilogy(time, np.abs(avg_ionic_charge_density - np.mean(equil_ionic_charge_density)), title="Relaxation of Ionic Charge Density", xlabel="Time (fs)", ylabel="|Ionic Charge Density - Equilibrium Ionic Charge Density| (e/Å^3)", output_file=output_prefix+"_relaxation_ionic_charge_density.png", xlim=xlim_timeseries)
    plot_semilogy(time, np.abs(avg_oxygen_xx_stress - np.mean(equil_oxygen_xx_stress)), title="Relaxation of Oxygen XX Stress", xlabel="Time (fs)", ylabel="|Oxygen XX Stress - Equilibrium Oxygen XX Stress|", output_file=output_prefix+"_relaxation_oxygen_xx_stress.png", xlim=xlim_timeseries)
    plot_semilogy(time, np.abs(avg_oxygen_yy_stress - np.mean(equil_oxygen_yy_stress)), title="Relaxation of Oxygen YY Stress", xlabel="Time (fs)", ylabel="|Oxygen YY Stress - Equilibrium Oxygen YY Stress|", output_file=output_prefix+"_relaxation_oxygen_yy_stress.png", xlim=xlim_timeseries)
    plot_semilogy(time, np.abs(avg_oxygen_zz_stress - np.mean(equil_oxygen_zz_stress)), title="Relaxation of Oxygen ZZ Stress", xlabel="Time (fs)", ylabel="|Oxygen ZZ Stress - Equilibrium Oxygen ZZ Stress|", output_file=output_prefix+"_relaxation_oxygen_zz_stress.png", xlim=xlim_timeseries)
    plot_semilogy(thermo_time, np.abs(avg_interfacial_temperature - np.mean(equil_interfacial_temperature)), title="Relaxation of Interfacial Temperature", xlabel="Time (fs)", ylabel="|Interfacial Temperature - Equilibrium Interfacial Temperature| (K)", output_file=output_prefix+"_relaxation_interfacial_temperature.png", xlim=xlim_timeseries)

# plot and save averaged densities
plot_time_series(time, avg_com_density, title="Average COM Density Over Time", xlabel="Time (fs)", ylabel="Density (molecules/Å^3)", output_file=output_prefix+"_avg_com_density.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_com_density) if equil_com_density is not None else None, benchmark_std=np.std(equil_com_density) if equil_com_density is not None else None, smoothed_data_fit=None)
plot_time_series(time, avg_potassium_density, title="Average Potassium Density Over Time", xlabel="Time (fs)", ylabel="Density (ions/Å^3)", output_file=output_prefix+"_avg_potassium_density.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_potassium_density) if equil_potassium_density is not None else None, benchmark_std=np.std(equil_potassium_density) if equil_potassium_density is not None else None, smoothed_data_fit=None)
plot_time_series(time, avg_chloride_density, title="Average Chloride Density Over Time", xlabel="Time (fs)", ylabel="Density (ions/Å^3)", output_file=output_prefix+"_avg_chloride_density.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_chloride_density) if equil_chloride_density is not None else None, benchmark_std=np.std(equil_chloride_density) if equil_chloride_density is not None else None, smoothed_data_fit=None)
plot_time_series(time, avg_ionic_charge_density, title="Average Ionic Charge Density Over Time", xlabel="Time (fs)", ylabel="Charge Density (e/Å^3)", output_file=output_prefix+"_avg_ionic_charge_density.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_ionic_charge_density) if equil_ionic_charge_density is not None else None, benchmark_std=np.std(equil_ionic_charge_density) if equil_ionic_charge_density is not None else None, smoothed_data_fit=None)
plot_time_series(time, avg_oxygen_xx_stress, title="Average Oxygen XX Stress Over Time", xlabel="Time (fs)", ylabel="Stress", output_file=output_prefix+"_avg_oxygen_xx_stress.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_oxygen_xx_stress) if equil_oxygen_xx_stress is not None else None, benchmark_std=np.std(equil_oxygen_xx_stress) if equil_oxygen_xx_stress is not None else None, smoothed_data_fit=None)
plot_time_series(time, avg_oxygen_yy_stress, title="Average Oxygen YY Stress Over Time", xlabel="Time (fs)", ylabel="Stress", output_file=output_prefix+"_avg_oxygen_yy_stress.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_oxygen_yy_stress) if equil_oxygen_yy_stress is not None else None, benchmark_std=np.std(equil_oxygen_yy_stress) if equil_oxygen_yy_stress is not None else None, smoothed_data_fit=None)
plot_time_series(time, avg_oxygen_zz_stress, title="Average Oxygen ZZ Stress Over Time", xlabel="Time (fs)", ylabel="Stress", output_file=output_prefix+"_avg_oxygen_zz_stress.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_oxygen_zz_stress) if equil_oxygen_zz_stress is not None else None, benchmark_std=np.std(equil_oxygen_zz_stress) if equil_oxygen_zz_stress is not None else None, smoothed_data_fit=None)
plot_time_series(thermo_time, avg_interfacial_temperature, title="Average Interfacial Temperature Over Time", xlabel="Time (fs)", ylabel="Temperature (K)", output_file=output_prefix+"_avg_interfacial_temperature.png", xlim=xlim_timeseries, benchmark_mean=np.mean(equil_interfacial_temperature) if equil_interfacial_temperature is not None else None, benchmark_std=np.std(equil_interfacial_temperature) if equil_interfacial_temperature is not None else None, smoothed_data_fit=None)  
save_timeseries_as_txt(time, avg_com_density, output_filename=output_prefix+"_avg_com_density.txt")
save_timeseries_as_txt(time, avg_potassium_density, output_filename=output_prefix+"_avg_potassium_density.txt")
save_timeseries_as_txt(time, avg_chloride_density, output_filename=output_prefix+"_avg_chloride_density.txt")
save_timeseries_as_txt(time, avg_ionic_charge_density, output_filename=output_prefix+"_avg_ionic_charge_density.txt")
save_timeseries_as_txt(time, avg_oxygen_xx_stress, output_filename=output_prefix+"_avg_oxygen_xx_stress.txt")
save_timeseries_as_txt(time, avg_oxygen_yy_stress, output_filename=output_prefix+"_avg_oxygen_yy_stress.txt")
save_timeseries_as_txt(time, avg_oxygen_zz_stress, output_filename=output_prefix+"_avg_oxygen_zz_stress.txt")
save_timeseries_as_txt(thermo_time, avg_interfacial_temperature, output_filename=output_prefix+"_avg_interfacial_temperature.txt")

if config['trajectory_type']=='process':
    # plot and save averaged transient density profiles
    plot_2d_heatmap(bin_centers, time, avg_com_transient_density.T, title="Average Transient COM Density Profile", xlabel="Z Position (Å)", ylabel="Time (fs)", output_file=output_prefix+"_avg_transient_com_density.png", xlim=(config['transient_density_profile']['zlo'], config['transient_density_profile']['zhi']), ylim=(None, None), colorbar_label="Density (molecules/Å^3)")
    plot_2d_heatmap(bin_centers, time, avg_potassium_transient_density.T, title="Average Transient Potassium Density Profile", xlabel="Z Position (Å)", ylabel="Time (fs)", output_file=output_prefix+"_avg_transient_potassium_density.png", xlim=(config['transient_density_profile']['zlo'], config['transient_density_profile']['zhi']), ylim=(None, None), colorbar_label="Density (ions/Å^3)")
    plot_2d_heatmap(bin_centers, time, avg_chloride_transient_density.T, title="Average Transient Chloride Density Profile", xlabel="Z Position (Å)", ylabel="Time (fs)", output_file=output_prefix+"_avg_transient_chloride_density.png", xlim=(config['transient_density_profile']['zlo'], config['transient_density_profile']['zhi']), ylim=(None, None), colorbar_label="Density (ions/Å^3)")

# plot hysteresis loops
if config['trajectory_type']=="process":
    plot_time_series(lower_wall_z_coordinates, avg_com_density, title="Hysteresis Loop: COM Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="COM Density (molecules/Å^3)", output_file=output_prefix+"_hysteresis_com_density.png", xlim=(config['plotting_options']['water_density']['hysteresis']['xlo'], config['plotting_options']['water_density']['hysteresis']['xhi']), ylim=(config['plotting_options']['water_density']['hysteresis']['ylo'], config['plotting_options']['water_density']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_potassium_density, title="Hysteresis Loop: Potassium Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Potassium Density (ions/Å^3)", output_file=output_prefix+"_hysteresis_potassium_density.png", xlim=(config['plotting_options']['potassium_density']['hysteresis']['xlo'], config['plotting_options']['potassium_density']['hysteresis']['xhi']), ylim=(config['plotting_options']['potassium_density']['hysteresis']['ylo'], config['plotting_options']['potassium_density']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_chloride_density, title="Hysteresis Loop: Chloride Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Chloride Density (ions/Å^3)", output_file=output_prefix+"_hysteresis_chloride_density.png", xlim=(config['plotting_options']['chloride_density']['hysteresis']['xlo'], config['plotting_options']['chloride_density']['hysteresis']['xhi']), ylim=(config['plotting_options']['chloride_density']['hysteresis']['ylo'], config['plotting_options']['chloride_density']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_ionic_charge_density, title="Hysteresis Loop: Ionic Charge Density vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Ionic Charge Density (e/Å^3)", output_file=output_prefix+"_hysteresis_ionic_charge_density.png", xlim=(config['plotting_options']['ionic_charge_density']['hysteresis']['xlo'], config['plotting_options']['ionic_charge_density']['hysteresis']['xhi']), ylim=(config['plotting_options']['ionic_charge_density']['hysteresis']['ylo'], config['plotting_options']['ionic_charge_density']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_oxygen_xx_stress, title="Hysteresis Loop: Oxygen XX Stress vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Oxygen XX Stress", output_file=output_prefix+"_hysteresis_oxygen_xx_stress.png", xlim=(config['plotting_options']['oxygen_xx_stress']['hysteresis']['xlo'], config['plotting_options']['oxygen_xx_stress']['hysteresis']['xhi']), ylim=(config['plotting_options']['oxygen_xx_stress']['hysteresis']['ylo'], config['plotting_options']['oxygen_xx_stress']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_oxygen_yy_stress, title="Hysteresis Loop: Oxygen YY Stress vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Oxygen YY Stress", output_file=output_prefix+"_hysteresis_oxygen_yy_stress.png", xlim=(config['plotting_options']['oxygen_yy_stress']['hysteresis']['xlo'], config['plotting_options']['oxygen_yy_stress']['hysteresis']['xhi']), ylim=(config['plotting_options']['oxygen_yy_stress']['hysteresis']['ylo'], config['plotting_options']['oxygen_yy_stress']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_oxygen_zz_stress, title="Hysteresis Loop: Oxygen ZZ Stress vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Oxygen ZZ Stress", output_file=output_prefix+"_hysteresis_oxygen_zz_stress.png", xlim=(config['plotting_options']['oxygen_zz_stress']['hysteresis']['xlo'], config['plotting_options']['oxygen_zz_stress']['hysteresis']['xhi']), ylim=(config['plotting_options']['oxygen_zz_stress']['hysteresis']['ylo'], config['plotting_options']['oxygen_zz_stress']['hysteresis']['yhi']))
    plot_time_series(lower_wall_z_coordinates, avg_interfacial_temperature[:-1], title="Hysteresis Loop: Interfacial Temperature vs Lower Wall Z", xlabel="Lower Wall Z (Å)", ylabel="Interfacial Temperature (K)", output_file=output_prefix+"_hysteresis_interfacial_temperature.png", xlim=(config['plotting_options']['interfacial_temperature']['hysteresis']['xlo'], config['plotting_options']['interfacial_temperature']['hysteresis']['xhi']), ylim=(config['plotting_options']['interfacial_temperature']['hysteresis']['ylo'], config['plotting_options']['interfacial_temperature']['hysteresis']['yhi']))