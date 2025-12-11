import os
import sys
import yaml
import numpy as np

# Get the absolute path of the directory containing the script (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (the parent of scripts/)
project_root = os.path.join(script_dir, '..')
# Add the project root to sys.path so Python can find the 'analysis' package
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.data_processing_tools import process_filename
from analysis.angular_analysis_tools import COM_trj, compute_local_basis_unit_vectors, arrange_trj_data_by_molecules, unwrap_trj
from analysis.transient_analysis_tools import plot_region_density_over_time, compute_velocity_distribution_in_region


# load configuration from YAML file
config_path = os.path.normpath(os.path.join(project_root, "configs", "20_example.yaml"))
with open(config_path) as f:
    config = yaml.safe_load(f)

base_path = config['base_path']
runs = config['runs']
files = config['files']
atom_types = config['atom_types']
dt = config['dt']

region = config['region']
density_config = config['density']
velocity_distribution_config = config['velocity_distribution']

output_config = config['output']

# create output base directory if it doesn't exist
out_folder = os.path.join(project_root, output_config['base_folder'])
os.makedirs(out_folder, exist_ok=True)

# Global arrays for averaging the results from individual runs
global_com_density = []
global_potassium_density = []
global_chloride_density = []

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
    trj, global2local, local2global, vel, force, stress, data, thermo_data = process_filename(trj_file,vel_file,force_file,stress_file,dat_file,thermo_file)

    # extract box size from data 
    box_size = [float(data["xhi"]) - float(data["xlo"]),
                float(data["yhi"]) - float(data["ylo"]),
                float(data["zhi"]) - float(data["zlo"])]

    # get basis vectors, positions of the oxygens, h1s and h2s
    a,b,c,positions_oxygens,positions_h1s,positions_h2s = compute_local_basis_unit_vectors(data, trj, atom_types["oxygen"], atom_types["hydrogen"], box_size, global2local=None, mode="debug")

    # arrange velocities and forces by water molecules
    velocities_oxygens, velocities_h1s, velocities_h2s = arrange_trj_data_by_molecules(data, vel, atom_types["oxygen"], atom_types["hydrogen"], global2local=global2local)
    forces_oxygens, forces_h1s, forces_h2s = arrange_trj_data_by_molecules(data, force, atom_types["oxygen"], atom_types["hydrogen"], global2local=global2local)

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
    time, com_density = plot_region_density_over_time(positions_COM, zlo=region["zlo"], zhi=region["zhi"], dt=dt, cross_sectional_area=box_size[0]*box_size[1])
    time, potassium_density = plot_region_density_over_time(positions_K, zlo=region["zlo"], zhi=region["zhi"], dt=dt, cross_sectional_area=box_size[0]*box_size[1])
    time, chloride_density = plot_region_density_over_time(positions_Cl, zlo=region["zlo"], zhi=region["zhi"], dt=dt, cross_sectional_area=box_size[0]*box_size[1])
    
    # collect densities for global averaging
    global_com_density.append(com_density)
    global_potassium_density.append(potassium_density)
    global_chloride_density.append(chloride_density)

# average densities over all runs
avg_com_density = np.mean(np.array(global_com_density), axis=0)
avg_potassium_density = np.mean(np.array(global_potassium_density), axis=0)
avg_chloride_density = np.mean(np.array(global_chloride_density), axis=0)
# standard deviation for error bars
std_com_density = np.std(np.array(global_com_density), axis=0)
std_potassium_density = np.std(np.array(global_potassium_density), axis=0)
std_chloride_density = np.std(np.array(global_chloride_density), axis=0)

