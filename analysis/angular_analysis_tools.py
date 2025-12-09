

# This function calculates the center of mass (COM) positions and velocities for water molecules
# WARNING: this assumes that there is a function that decomposes the entire trajectory data into just the data of the oxygens and the hydrogens of water.
def COM_trj(positions_oxygens,positions_h1s,positions_h2s,velocities_oxygens,velocities_h1s,velocities_h2s,data,hydrogen_type,oxygen_type):
    hydrogen_mass=data["Masses"][hydrogen_type]
    oxygen_mass=data["Masses"][oxygen_type]

    positions_COM=(positions_oxygens*oxygen_mass+positions_h1s*hydrogen_mass+positions_h2s*hydrogen_mass)/(oxygen_mass+2*hydrogen_mass)
    velocities_COM=(velocities_oxygens*oxygen_mass+velocities_h1s*hydrogen_mass+velocities_h2s*hydrogen_mass)/(oxygen_mass+2*hydrogen_mass)

    return positions_COM, velocities_COM