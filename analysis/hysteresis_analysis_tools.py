import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def wavenumber_2_period_amplitude(wavenumber, temperature=300):
    # move this import here when you make this into an actual function file
    import numpy as np

    c = 2.99792E-5 # speed of light in cm/fs
    k_b = 8.314654E-7 # g*Ang^2/(fs^2*K*mol)
    m = 12.0107 # g/mol

    freq = wavenumber * c # in 1/fs
    period = 1 / freq # in fs

    amplitude = np.sqrt((k_b*temperature)/(m*freq**2*2*np.pi**2)) # in Angstroms

    return period, amplitude

def compute_phase_shift(x_property: np.array, y_property: np.array, time: np.array, number_of_phase_bins: Optional[int] = 500) -> tuple:

    sign_x_property = np.sign(x_property)
    
    zero_crossings_x_property = np.where(sign_x_property[:-1] * sign_x_property[1:] < 0)[0]
    dx_property_dt = np.gradient(x_property, time)
    valid_zero_crossings_x_property = zero_crossings_x_property[dx_property_dt[zero_crossings_x_property] > 0]

    if 0 not in valid_zero_crossings_x_property:
        valid_zero_crossings_x_property = np.insert(valid_zero_crossings_x_property, 0, 0)
    
    phase_grid = np.linspace(0, 2*np.pi, number_of_phase_bins)
    x_property_phase_shift = []
    y_property_phase_shift = []

    for i in range(len(valid_zero_crossings_x_property)-1):
        start_idx = valid_zero_crossings_x_property[i]
        end_idx = valid_zero_crossings_x_property[i+1]

        x_segment = x_property[start_idx:end_idx]
        y_segment = y_property[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]

        segment_phase = 2 * np.pi * (time_segment - time_segment[0]) / (time_segment[-1] - time_segment[0])

        x_interp = np.interp(phase_grid, segment_phase, x_segment)
        y_interp = np.interp(phase_grid, segment_phase, y_segment)

        x_property_phase_shift.append(x_interp)
        y_property_phase_shift.append(y_interp)

    x_property_phase_shift = np.array(x_property_phase_shift)
    y_property_phase_shift = np.array(y_property_phase_shift)
    
    return x_property_phase_shift, y_property_phase_shift




