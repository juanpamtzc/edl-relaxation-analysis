import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Optional

# Left to improve: 
# - add error handling
# - add unit tests
def unwrap_trj(positions_oxygens: np.array, positions_h1s: np.array, positions_h2s: np.array, data: dict) -> tuple:
    """
    Unwraps the positions of the oxygens and hydrogens based on the positions of the oxygens of each water molecule using the minimum image convention and the box size
    This allows for the proper computation of bond lengths, as bonds no longer wrap at periodic boundary conditions.
    The assumption is that there are periodic boundary conditions in all dimensions.

    Args:
        positions_oxygens:  a numpy array with shape (M,N,3) with the positions of the oxygen of each water molecule, where M is the number of timeframes,
                            N is the number of molecules, and 3 is for x,y,z.
        positions_h1s:      a numpy array with shape (M,N,3) with the positionf of one of the hydrogens of each water molecule, the molecule indeces
                            agree with the molecule indeces from positions_oxygens.
        positions_h2s:      a numpy array with shape (M,N,3) with the positions of one of other hydrogen of each water molecule.
        data:               a dictionary with the information from the data file from the corresponding simulation as given by readDatFile in reading_tools.py.
    
    Returns:
        positions_oxygens:  same numpy array that was input.
        positions_h1s:      same numpy array that was input, but with the hydrogen positions unwrapped relative to their corresponding oxygen using the minimum
                            image convention.
        positions_h2s:      same numpy array that was input, but with the hydrogen positions unwrapped relative to their corresponding oxygen using the minimum
                            image convention.
    """
   
    box_size=[np.double(data["xhi"])-np.double(data["xlo"]),
              np.double(data["yhi"])-np.double(data["ylo"]),
              np.double(data["zhi"])-np.double(data["zlo"])]
    
    positions_h1s = positions_h1s - np.round((positions_h1s - positions_oxygens) / box_size) * box_size
    positions_h2s = positions_h2s - np.round((positions_h2s - positions_oxygens) / box_size) * box_size

    return positions_oxygens, positions_h1s, positions_h2s

# Left to improve:
# - add error handling
# - add unit tests
def arrange_trj_data_by_molecules(data: dict, trj: np.array, oxygen_type: int, hydrogen_type: int, global2local: Optional[dict] = None) -> tuple:
    """
    Arranges the trajectory data from a simulation containing water molecules (could also contain other sorts of atoms/molecules) by water molecule.
    Since trajectory data usually has less information regarding molecules than data file data, the function requires the data from the corresponding data file 
    to group the trajectory data by water molecule. This function can be used with any sort of trajectory data as long as it is given as a numpy array with 
    shape (M,N,4), where M is the number of timeframes, N is the number of atoms, and 4 is for atom_type, x, y, z.

    Args:
        data:               dictionary with the information from the data file from the corresponding simulation as given by readDatFile in reading_tools.py.
        trj:                numpy array with shape (M,N,4) where M is the number of timeframes, N is the number of atoms, and the last dimension corresponds to (atom type, x, y, z), 
                            where atom type is an integer.
        oxygen_type:        integer corresponding to the atom type of the oxygen atoms in the water molecules.
        hydrogen_type:      integer corresponding to the atom type of the hydrogen atoms in the water molecules.
        global2local:       (optional) dictionary mapping global atom indices to local atom indices in the trajectory data. If provided, only water molecules with all atoms present 
                            in the trajectory will be included. This allows for handling trajectory data that may not include all atoms from the original data file.
    
    Returns:
        oxygens:            numpy array with shape (M, N_water_molecules, 3) with the positions of the oxygen atoms of each water molecule, where M is the number of timeframes,
                            N_water_molecules is the number of water molecules, and 3 is for x, y, z.
        h1s:                numpy array with shape (M, N_water_molecules, 3) with the positions of one of the hydrogen atoms of each water molecule.
        h2s:                numpy array with shape (M, N_water_molecules, 3) with the positions of the other hydrogen atom of each water molecule.

    NOTE: The positions are might be wrapped at periodic boundary conditions. Use unwrap_trj to unwrap them if needed.
    """

    # Create mappings
    atom_2_molecule = {}
    OXYGENS = {}
    H1S = {}
    H2S = {}
    N_water_molecules = 0

    for atom in data["Atoms"]:
        atom_2_molecule[atom] = data["Atoms"][atom]["molecule ID"]
        if data["Atoms"][atom]["atom type"] == oxygen_type:
            N_water_molecules += 1
            OXYGENS[atom_2_molecule[atom]] = atom
        if data["Atoms"][atom]["atom type"] == hydrogen_type:
            if atom_2_molecule[atom] not in H1S:
                H1S[atom_2_molecule[atom]] = atom
            elif atom != H1S[atom_2_molecule[atom]]:
                H2S[atom_2_molecule[atom]] = atom

    # remake OXYGENS, H1S, and H2S to only include the molecules that have at least one atom in the global2local directory
    if not global2local==None:
        new_molecule_ID=0
        old2new_moleculeID_map={}
        newOXYGENS={}
        newH1S={}
        newH2S={}
        for molecule in OXYGENS:
            oxygen = OXYGENS[molecule]
            h1 = H1S[molecule]
            h2 = H2S[molecule]
            if (oxygen in global2local) and (h1 in global2local) and (h2 in global2local):
                new_molecule_ID+=1
                old2new_moleculeID_map[molecule]=new_molecule_ID
                newOXYGENS[new_molecule_ID]=global2local[OXYGENS[molecule]]
                newH1S[new_molecule_ID]=global2local[H1S[molecule]]
                newH2S[new_molecule_ID]=global2local[H2S[molecule]]
        N_water_molecules=new_molecule_ID
        OXYGENS=newOXYGENS
        H1S=newH1S
        H2S=newH2S

    # Get dimensions
    M = trj.shape[0]

    # Initialize arrays
    oxygens = np.zeros((M, N_water_molecules, 3))
    h1s = np.zeros((M, N_water_molecules, 3))
    h2s = np.zeros((M, N_water_molecules, 3))

    # Populate arrays based on molecule indexing
    for mol_id in range(1, N_water_molecules + 1):  # Assuming molecule IDs start at 1
        if mol_id in OXYGENS:
            oxygens[:, mol_id - 1, :] = np.double(trj[:, OXYGENS[mol_id] - 1, 1:])
        if mol_id in H1S:
            h1s[:, mol_id - 1, :] = np.double(trj[:, H1S[mol_id] - 1, 1:])
        if mol_id in H2S:
            h2s[:, mol_id - 1, :] = np.double(trj[:, H2S[mol_id] - 1, 1:])
        
    return oxygens, h1s, h2s

# This function calculates the center of mass (COM) positions and velocities for water molecules
# WARNING: this assumes that there is a function that decomposes the entire trajectory data into just the data of the oxygens and the hydrogens of water.
def COM_trj(positions_oxygens,positions_h1s,positions_h2s,velocities_oxygens,velocities_h1s,velocities_h2s,data,hydrogen_type,oxygen_type):
    hydrogen_mass=data["Masses"][hydrogen_type]
    oxygen_mass=data["Masses"][oxygen_type]

    positions_COM=(positions_oxygens*oxygen_mass+positions_h1s*hydrogen_mass+positions_h2s*hydrogen_mass)/(oxygen_mass+2*hydrogen_mass)
    velocities_COM=(velocities_oxygens*oxygen_mass+velocities_h1s*hydrogen_mass+velocities_h2s*hydrogen_mass)/(oxygen_mass+2*hydrogen_mass)

    return positions_COM, velocities_COM

def compute_local_basis_unit_vectors(data, trj, oxygen_type, hydrogen_type, box_size, global2local=None, mode="debug"):

    # use arrange_trj_data_by_molecules to get the positions of the oxygens and hydrogens
    # refer to the documentation and comments of that function (it's in this same script) for more details
    oxygens, h1s, h2s = arrange_trj_data_by_molecules(data,trj,oxygen_type,hydrogen_type,global2local=global2local)

    # Find the vectors from each hydrogen to its corresponding oxygen, applying minimum image convention (most simulations will have periodic boundary conditions, and if the boundary conditions are not periodic it won't affect the behavior of the code)
    r_h1_rel = h1s - oxygens - np.round((h1s - oxygens) / box_size) * box_size
    r_h2_rel = h2s - oxygens - np.round((h2s - oxygens) / box_size) * box_size

    if mode=="debug":
        print("maximum oh1 length:")
        print(np.max(np.linalg.norm(r_h1_rel,axis=2)))
        print("maximum oh2 length:")
        print(np.max(np.linalg.norm(r_h2_rel,axis=2)))

    # find the local basis unit vectors for each water molecule:
    #   (a) is parallel to the dipole vector, going from the oxygen to betwenn both hydrogen
    #   (b) is perpendicular to (a) and in the direction of the normal vector of the molecular plane (uses vector from oxygen to one of the two hydrogens to define the molecular plane)
    #   (c) is perpendicular to both (a) and (b)
    a = r_h1_rel+r_h2_rel
    a = np.divide(a, np.linalg.norm(a,axis=2,keepdims=True), where=np.linalg.norm(a,axis=2,keepdims=True) != 0)  # Avoid division by zero (set to 0 where norm is zero)

    b = np.cross(r_h1_rel,a,axis=2)
    b = np.divide(b, np.linalg.norm(b,axis=2,keepdims=True), where=np.linalg.norm(b,axis=2,keepdims=True) != 0)  # Avoid division by zero (set to 0 where norm is zero)

    c = np.cross(a,b,axis=2)
    c = np.divide(c, np.linalg.norm(c,axis=2,keepdims=True), where=np.linalg.norm(c,axis=2,keepdims=True) != 0)  # Avoid division by zero (set to 0 where norm is zero)

    # include a debug statement to check that the unit vectors are properly normalized and orthogonal
    # CHECKS
    if mode=="debug":
        print("a:")
        print(a[0,0,:])
        print("norm(a):\t***should be within machine precision of 1***")
        print(np.linalg.norm(a[0,0,:]))
        print("b:")
        print(b[0,0,:])
        print("norm(b):")
        print(np.linalg.norm(b[0,0,:]))
        print("c:")
        print(c[0,0,:])
        print("norm(c):\t***should be within machine precision of 1***")
        print(np.linalg.norm(c[0,0,:]))
        print("a dot b:\t***should be within machine precision of 0***")
        print(np.dot(a[0,0,:],b[0,0,:]))
        print("b dot c:\t***should be within machine precision of 0***")
        print(np.dot(b[0,0,:],c[0,0,:]))
        print("c dot a:\t***should be within machine precision of 0***")
        print(np.dot(a[0,0,:],c[0,0,:]))

    return a, b, c, oxygens, h1s, h2s

# This function computes the time derivative of the local basis vectors a, b, c
# It uses the central difference method by default, but can also use forward or backward difference methods
def compute_de_dt(a,b,c,dt,style="central difference"):
    M=a.shape[0]
    da_dt=np.zeros(a.shape)
    db_dt=np.zeros(b.shape)
    dc_dt=np.zeros(c.shape)
    if style=="backward difference":
        da_dt[1:,:,:]=(a[1:,:,:]-a[:M-1,:,:])/dt
        db_dt[1:,:,:]=(b[1:,:,:]-b[:M-1,:,:])/dt
        dc_dt[1:,:,:]=(c[1:,:,:]-c[:M-1,:,:])/dt
    elif style=="forward difference":
        da_dt[:M-1,:,:]=(a[1:,:,:]-a[:M-1,:,:])/dt
        db_dt[:M-1,:,:]=(b[1:,:,:]-b[:M-1,:,:])/dt
        dc_dt[:M-1,:,:]=(c[1:,:,:]-c[:M-1,:,:])/dt
    elif style=="central difference":
        da_dt[1:M-1,:,:]=(a[2:,:,:]-a[:M-2,:,:])/(2*dt)
        db_dt[1:M-1,:,:]=(b[2:,:,:]-b[:M-2,:,:])/(2*dt)
        dc_dt[1:M-1,:,:]=(c[2:,:,:]-c[:M-2,:,:])/(2*dt)
    return da_dt, db_dt, dc_dt

# This function computes the angular velocity of the molecule based on the time derivative of the local basis vectors
def compute_angular_velocity_from_basis_vectors(a,b,c,da_dt,db_dt,dc_dt,style="central difference"):
    M=a.shape[0]
    angular_velocities=np.zeros(a.shape)
    angular_velocities[:,:,0]=np.sum(db_dt*c,axis=2)
    angular_velocities[:,:,1]=np.sum(dc_dt*a,axis=2)
    angular_velocities[:,:,2]=np.sum(da_dt*b,axis=2)
    if style=="backward difference":
        angular_velocities=angular_velocities[1:,:,:]
    elif style=="forward difference":
        angular_velocities=angular_velocities[:M-1,:,:]
    elif style=="central difference":
        angular_velocities=angular_velocities[1:M-1,:,:]
    return angular_velocities

# This function transforms the angular velocities from the local frame (a,b,c) to the lab frame (x,y,z)
def transform_angular_velocities_to_lab_frame(angular_velocities, a, b, c, style="central difference"):
    # Build rotation matrices R to go from local (a,b,c) to lab (x,y,z)
    R = np.stack([a, b, c], axis=-1)  # shape (T, N, 3, 3)

    if style == "central difference":
        R = R[1:-1]  # match time steps with angular_velocities
    elif style == "backward difference":
        R = R[1:]
    elif style == "forward difference":
        R = R[:-1]

    # Rotate local angular velocities to lab frame
    angular_velocities_global = np.einsum('...ij,...j->...i', R, angular_velocities)
    return angular_velocities_global

# This function computes the time- and space-averaged angular distribution between local basis vectors a, b, c and the global z-axis
def compute_angle_distribution_with_z(a, b, c, nbins=180, output_file=None, plot=False):
    """
    Compute the time- and space-averaged angular distribution between local basis vectors
    a, b, c and the global z-axis. Optionally save and/or plot the result.

    Parameters:
    - a, b, c: Arrays of shape (T, N, 3), basis vectors over time and molecules
    - nbins: Number of bins for histogram (default: 180 for 1-degree resolution)
    - output_file: Path to save the output .txt file (optional)
    - plot: If True, display a plot of the distributions

    Returns:
    - bin_centers: Midpoints of angle bins (in degrees)
    - hist_a, hist_b, hist_c: Normalized histograms for vectors a, b, and c
    """

    def angle_with_z(vectors):
        z = np.array([0, 0, 1])
        cos_theta = np.clip(np.einsum('...i,i->...', vectors, z), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))  # angles in degrees

    # Flatten time and molecule axes
    angles_a = angle_with_z(a.reshape(-1, 3))
    angles_b = angle_with_z(b.reshape(-1, 3))
    angles_c = angle_with_z(c.reshape(-1, 3))

    # Histogram for each set of angles
    hist_a, bin_edges = np.histogram(angles_a, bins=nbins, range=(0, 180), density=True)
    hist_b, _         = np.histogram(angles_b, bins=nbins, range=(0, 180), density=True)
    hist_c, _         = np.histogram(angles_c, bins=nbins, range=(0, 180), density=True)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Optionally save to .txt file
    if output_file is not None:
        header = "Angle_Degrees\tPDF_a\tPDF_b\tPDF_c"
        output_data = np.column_stack((bin_centers, hist_a, hist_b, hist_c))
        np.savetxt(output_file, output_data, fmt="%.6f", delimiter="\t", header=header)

    # Optionally plot
    if plot:
        plt.figure(figsize=(7,5))
        plt.plot(bin_centers, hist_a, label='a • z', lw=1.5)
        plt.plot(bin_centers, hist_b, label='b • z', lw=1.5)
        plt.plot(bin_centers, hist_c, label='c • z', lw=1.5)
        plt.xlabel("Angle with z-axis (degrees)")
        plt.ylabel("Probability Density")
        plt.title("Angular Distribution with z-axis")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return bin_centers, hist_a, hist_b, hist_c