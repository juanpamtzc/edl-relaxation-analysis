from analysis.reading_tools import readTRJFile, readDatFile, readTRJFile_stresses, readLAMMPSThermodynamicFile
from typing import Optional
import numpy as np

def process_filename(trj_filename: str, vel_filename: str, force_filename: str, stress_filename: str, dat_filename: str, thermo_filename: str, thermo_process_ID: Optional[int] = 1) -> tuple:
    """
    Processes the provided filenames for trajectory, velocity, force, stress, data, and thermodynamic output files.

    Args:
        trj_filename:           The filename for the trajectory data.
        vel_filename:           The filename for the velocity data.
        force_filename:         The filename for the force data.
        stress_filename:        The filename for the stress data.
        dat_filename:           The filename for the data file. 
        thermo_filename:        The filename for the thermodynamic data.
        thermo_process_ID (Optional[int]): The process ID for thermodynamic data processing. Default is 1.
    
    Returns:
        trj:                    A numpy array shaped (M, N, 3) containing trajectory data.
        global2local:           A dictionary mapping global atom IDs to local atom IDs.
        local2global:           A dictionary mapping local atom IDs to global atom IDs.
        vel:                    A numpy array shaped (M, N, 3) containing velocity data.
        force:                  A numpy array shaped (M, N, 3) containing force data.
        stress:                 A numpy array shaped (M, N, 6) containing stress data.
        data:                   A dictionary containing data from the .dat file.
        thermodynamic_data:     A pandas dataframe containing thermodynamic data.
    """

    trj, local2global, global2local=readTRJFile(trj_filename, True, None)
    vel=readTRJFile(vel_filename, True, None)[0]
    force=readTRJFile(force_filename, True, None)[0]
    stress=readTRJFile_stresses(stress_filename, True, None)[0]
    data=readDatFile(dat_filename)
    thermodynamic_data=readLAMMPSThermodynamicFile(thermo_filename, thermo_process_ID)
    return trj, global2local, local2global, vel, force, stress, data, thermodynamic_data

def save_timeseries_as_txt(time: np.array, data: np.array, output_filename: str):
    """
    Saves the provided time series data to a text file.

    Args:
        time:               A numpy array containing time values.
        data:               A numpy array containing corresponding data values.
        output_filename:    The filename for the output text file.
    """
    combined_data = np.column_stack((time, data))
    np.savetxt(output_filename, combined_data, header="Time\tData")

def read_timeseries_from_txt(filename: str) -> tuple:
    """
    Reads time series data from a text file.

    Args:
        filename:   The filename of the text file containing time series data.
    
    Returns:
        A tuple containing two numpy arrays: time values and corresponding data values.
    """
    data = np.loadtxt(filename, skiprows=1)
    time = data[:, 0]
    values = data[:, 1:]
    return time, values