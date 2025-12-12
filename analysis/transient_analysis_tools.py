import numpy as np
from analysis.plotting_tools import plot_time_series, plot_distribution_function
# delete once no longer needed
import matplotlib.pyplot as plt

# This function finds the z-coordinates of a representative carbon atom forming the lower wall over time
def find_lower_wall_z_coordinates(trj, carbon_type, low_z):

    # make a mask for carbon atoms below low_z
    representative_carbon_mask = (trj[0,:,0]==carbon_type) & (trj[0,:,3]<low_z)
    # get the index of the first representative carbon
    representative_carbon_index = np.argmax(representative_carbon_mask)
    # extract the z-coordinates of that representative carbon over time
    representative_carbon_z_coordinates = trj[:,representative_carbon_index,3]

    return representative_carbon_z_coordinates

# This function computes the average property (velocity, force, etc.) inside a specified region along the z-axis
# FUTURE WORK: Add support for regions inside a region that does not span the x-y plane
def compute_average_property_over_time(positions, property_array, zlo=0.0, zhi=20.0, dt=1.0,component=None,plot=True,plot_prefix="average_property_vs_time"):

    # Extract z coordinates
    z_coordinates = positions[:, :, 2]

    # Boolean mask of atoms inside region at each timestep
    inside_region_mask = (z_coordinates >= zlo) & (z_coordinates < zhi)

    # Extract property values for atoms in the region
    if component is None:
        # magnitude of vector
        property_desired_components = np.linalg.norm(property_array, axis=2)
    else:
        property_desired_components = property_array[:, :, component]

    # Mask out atoms outside the region
    property_desired_components_inside_region = np.where(inside_region_mask, property_desired_components, np.nan)

    # Average over atoms for each timestep (ignoring NaNs)
    avg_over_time = np.nanmean(property_desired_components_inside_region, axis=1)

    # Time axis
    n_steps = positions.shape[0]
    time = np.arange(n_steps) * dt

    if plot:
        # Plot
        plot_time_series(time, avg_over_time,
                         title=f"Average Property in Region [{zlo}, {zhi}] vs Time",
                         xlabel="Time",
                         ylabel="Average Property",
                         output_file=f"{plot_prefix}.png")
        # DELETE IF NO LONGER NEEDED
        #plt.figure(figsize=(8, 6))
        #plt.plot(time, avg_over_time)
        #plt.xlabel("Time")
        #plt.ylabel("Average Property")
        #plt.title(f"Average Property in Region [{zlo}, {zhi}] vs Time")
        #plt.grid(True)
        #plt.tight_layout()
        #plt.savefig(f"{plot_prefix}.png")
        #plt.close()

    return time, avg_over_time

# This function computes the probability distribution of velocities for atoms inside a specified region along the z-axis
# This can help in assessing whether or not the use of an equilibrium thermostat in that region is appropriate
# FUTURE WORK: Add support for regions inside a region that does not span the x-y plane
def compute_velocity_distribution_in_region(positions, velocities, zlo=0.0, zhi=20.0, component=None, bins=50, plot=True, plot_prefix="velocity_distribution"):

    # Extract z coordinates
    z_coordinates = positions[:, :, 2]
    
    # Boolean mask of atoms inside region at each timestep
    inside_region_mask = (z_coordinates >= zlo) & (z_coordinates < zhi)
    
    # Extract velocity values
    if component is None:
        # Speed (magnitude of velocity vector)
        velocities_desired_components = np.linalg.norm(velocities, axis=2)
    else:
        velocities_desired_components = velocities[:, :, component]
    
    # Flatten and extract only velocities of atoms inside the region
    # This is efficient: we flatten both arrays and use the flattened mask
    velocities_desired_components_inside_region = velocities_desired_components[inside_region_mask]
    
    if plot:
        # Plot histogram (probability distribution) of velocities
        plt.figure(figsize=(8, 6))
        counts, bins, _ = plt.hist(velocities_desired_components_inside_region, bins=bins, density=True, 
                                    alpha=0.7, edgecolor='black')
        
        # Labels
        if component is None:
            plt.xlabel("Speed")
            plt.ylabel("Probability Density")
            plt.title(f"Speed Distribution in Region [{zlo}, {zhi}]")
        else:
            comp_label = ['vx', 'vy', 'vz'][component]
            plt.xlabel(f"Velocity Component {comp_label}")
            plt.ylabel("Probability Density")
            plt.title(f"{comp_label} Distribution in Region [{zlo}, {zhi}]")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}.png", dpi=150)
        plt.close()
    
    return bins, counts

# This function computes the average (over all molecules in a region) cosine of the angle between the molecular local unit basis vectors and the z-axis as a function of time
def compute_average_cos_over_time(positions,cos_array,zlo=0.0,zhi=20.0,dt=1.0,plot=True,plot_prefix="average_cos_vs_time"):

    # Extract z-coordinates
    z_coordinates = positions[:, :, 2]

    # Boolean mask selecting atoms inside region at each timestep
    inside_region_mask = (z_coordinates >= zlo) & (z_coordinates < zhi)

    # Mask cosine values outside region (→ NaN)
    cosine_inside_region = np.where(inside_region_mask, cos_array, np.nan)

    # Average cosine per timestep, ignoring NaNs
    avg_cosine_inside_region = np.nanmean(cosine_inside_region, axis=1)

    # Time axis
    n_steps = positions.shape[0]
    time = np.arange(n_steps) * dt

    if plot:
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(time, avg_cosine_inside_region)
        plt.xlabel("Time")
        plt.ylabel("Average cos(θ)")
        plt.title(f"Average cos(θ) in Region [{zlo}, {zhi}] vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}.png")
        plt.close()
    
    return time, avg_cosine_inside_region

# This function computes and plots the average (over all water molecules) translational and rotational kinetic energy components as a function of time
def compute_water_kinetic_energy(com_positions, com_velocities, angular_velocities, plot=True, plot_prefix='water_kinetic_energy', style="central difference"):
   
    n_timesteps, n_molecules, _ = com_positions.shape
    time = np.arange(n_timesteps)

    translational_ke = 0.5 * np.sum(com_velocities**2, axis=2)  # shape: (T, M)
    translational_ke_x = 0.5 * com_velocities[:,:,0]**2
    translational_ke_y = 0.5 * com_velocities[:,:,1]**2
    translational_ke_z = 0.5 * com_velocities[:,:,2]**2
    rotational_ke = 0.5 * np.sum(angular_velocities**2, axis=2)  # shape: (T, M)
    rotational_ke_a = 0.5 * angular_velocities[:,:,0]**2
    rotational_ke_b = 0.5 * angular_velocities[:,:,1]**2
    rotational_ke_c = 0.5 * angular_velocities[:,:,2]**2

    average_translational_ke = np.mean(translational_ke, axis=1)  # shape: (T,)
    average_translational_ke_x = np.mean(translational_ke_x, axis=1)
    average_translational_ke_y = np.mean(translational_ke_y, axis=1)
    average_translational_ke_z = np.mean(translational_ke_z, axis=1)
    average_rotational_ke = np.mean(rotational_ke, axis=1)  # shape: (T,)
    average_rotational_ke_a = np.mean(rotational_ke_a, axis=1)
    average_rotational_ke_b = np.mean(rotational_ke_b, axis=1)
    average_rotational_ke_c = np.mean(rotational_ke_c, axis=1)

    average_translational_ke = np.mean(translational_ke, axis=1)/average_translational_ke[0]  # shape: (T,)
    average_translational_ke_x = np.mean(translational_ke_x, axis=1)/average_translational_ke_x[0]
    average_translational_ke_y = np.mean(translational_ke_y, axis=1)/average_translational_ke_y[0]
    average_translational_ke_z = np.mean(translational_ke_z, axis=1)/average_translational_ke_z[0]
    average_rotational_ke = np.mean(rotational_ke, axis=1)/average_rotational_ke[0]  # shape: (T,)
    average_rotational_ke_a = np.mean(rotational_ke_a, axis=1)/average_rotational_ke_a[0]
    average_rotational_ke_b = np.mean(rotational_ke_b, axis=1)/average_rotational_ke_b[0]
    average_rotational_ke_c = np.mean(rotational_ke_c, axis=1)/average_rotational_ke_c[0]

    
    if style == "central difference":
        time= time[1:-1]
        average_translational_ke = average_translational_ke[1:-1]
        average_translational_ke_x = average_translational_ke_x[1:-1]
        average_translational_ke_y = average_translational_ke_y[1:-1]
        average_translational_ke_z = average_translational_ke_z[1:-1]
    elif style == "forward difference" or style == "backward difference":
        time= time[:-1]
        average_translational_ke = average_translational_ke[:-1]
        average_translational_ke_x = average_translational_ke_x[:-1]
        average_translational_ke_y = average_translational_ke_y[:-1]
        average_translational_ke_z = average_translational_ke_z[:-1]

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(time, average_translational_ke, label='Translational KE', color='blue')
        plt.plot(time, average_translational_ke_x, label='Translational KE X', color='cyan', linestyle=':')
        plt.plot(time, average_translational_ke_y, label='Translational KE Y', color='dodgerblue', linestyle='--')
        plt.plot(time, average_translational_ke_z, label='Translational KE Z', color='navy', linestyle='-.')
        plt.plot(time, average_rotational_ke, label='Rotational KE', color='orange')
        plt.plot(time, average_rotational_ke_a, label='Rotational KE A', color='gold', linestyle=':')
        plt.plot(time, average_rotational_ke_b, label='Rotational KE B', color='darkorange', linestyle='--')
        plt.plot(time, average_rotational_ke_c, label='Rotational KE C', color='orangered', linestyle='-.')
        plt.xlabel('Time')
        plt.ylabel('Kinetic Energy')
        plt.title('Average Kinetic Energy of Water Molecules over Time')
        plt.legend()
        plt.grid()
        plt.savefig(f'{plot_prefix}_kinetic_energy.png')
        plt.close()
    
    return time, average_translational_ke, average_translational_ke_x, average_translational_ke_y, average_translational_ke_z, average_rotational_ke, average_rotational_ke_a, average_rotational_ke_b, average_rotational_ke_c
    
# This function computes the number density of atoms inside a specified region along the z-axis as a function of time
def compute_region_density_over_time(positions, zlo=0.0, zhi=7.5, dt=1.0, cross_sectional_area=1.0, plot=True, plot_prefix="density_vs_time"):

    # unpack
    n_steps = positions.shape[0]
    height = zhi - zlo

    # compute number of atoms in region at each timestep
    # vectorized boolean mask
    z_coordinates = positions[:, :, 2]
    inside = (z_coordinates >= zlo) & (z_coordinates < zhi)
    counts = inside.sum(axis=1)

    # number density = N / volume
    volume = cross_sectional_area * height
    density = counts / volume

    # time axis
    time = np.arange(n_steps) * dt

    if plot:
        # plot
        plt.figure(figsize=(8, 6))
        plt.plot(time, density)
        plt.xlabel("Time")
        plt.ylabel("Number Density (atoms / volume)")
        plt.title(f"Number Density in Region [{zlo}, {zhi}] vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}.png")
        plt.close()

    return time, density