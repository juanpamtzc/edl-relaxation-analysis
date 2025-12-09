import numpy as np
import matplotlib.pyplot as plt

# This function plots the average property (velocity, force, etc.) inside a specified region along the z-axis
# FUTURE WORK: Add support for regions inside a region that does not span the x-y plane
def plot_average_property_over_time(positions, property_array, zlo=0.0, zhi=20.0, dt=1.0,component=None,plot_prefix="average_property_vs_time"):

    # Extract z coordinates
    zvals = positions[:, :, 2]

    # Boolean mask of atoms inside region at each timestep
    inside = (zvals >= zlo) & (zvals < zhi)

    # Extract property values for atoms in the region
    if component is None:
        # magnitude of vector
        prop_vals = np.linalg.norm(property_array, axis=2)
    else:
        prop_vals = property_array[:, :, component]

    # Mask out atoms outside the region
    masked_vals = np.where(inside, prop_vals, np.nan)

    # Average over atoms for each timestep (ignoring NaNs)
    avg_over_time = np.nanmean(masked_vals, axis=1)

    # Time axis
    n_steps = positions.shape[0]
    time = np.arange(n_steps) * dt

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(time, avg_over_time)
    plt.xlabel("Time")
    plt.ylabel("Average Property")
    plt.title(f"Average Property in Region [{zlo}, {zhi}] vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.png")
    plt.close()