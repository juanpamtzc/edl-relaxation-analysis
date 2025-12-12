from analysis.reading_tools import readTRJFile, readDatFile, readTRJFile_stresses, readLAMMPSThermodynamicFile

# This function processes corresponding data, trajectory, velocity, force, stress, and thermodynamic output files and returns their contents for analysis
def process_filename(trj_filename, vel_filename, force_filename, stress_filename, dat_filename, thermo_filename, thermo_process_ID=1):
    trj, local2global, global2local=readTRJFile(trj_filename, True, None)
    vel=readTRJFile(vel_filename, True, None)[0]
    force=readTRJFile(force_filename, True, None)[0]
    stress=readTRJFile_stresses(stress_filename, True, None)[0]
    data=readDatFile(dat_filename)
    thermodynamic_data=readLAMMPSThermodynamicFile(thermo_filename, thermo_process_ID)
    return trj, global2local, local2global, vel, force, stress, data, thermodynamic_data