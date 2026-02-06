import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman
from typing import Optional
import matplotlib.pyplot as plt


def compute_avg_d_dt(property: np.array, time: np.array) -> float:
    """
    Compute the average time derivative of a property over the entire time series.

    Parameters
    ----------
    property : np.array
        The property values as a function of time.
    time : np.array
        The time values corresponding to the property.

    Returns
    -------
    float
        The average time derivative of the property.
    """
    d_property_dt = np.gradient(property, time)
    avg_d_property_dt = np.mean(d_property_dt)

    return avg_d_property_dt

def compute_phase_difference(A: np.array, B: np.array, time:np.array, threshold_percentage: Optional[float] = 1E-4) -> np.array:
    """
    Compute the phase difference between two properties A and B as a function of time.

    Parameters
    ----------
    A : np.array
        The first property values as a function of time.
    B : np.array
        The second property values as a function of time.
    time : np.array
        The time values corresponding to the properties.

    Returns
    -------
    np.array
        The phase difference between properties A and B as a function of time.
    """
    
    N = len(time)
    dt = time[1] - time[0]

    A_fft = fft(A)
    B_fft = fft(B)

    window = blackman(N)
    A_windowed_fft = fft(A * window)
    B_windowed_fft = fft(B * window)

    threshold_A = threshold_percentage * np.max(np.abs(A_windowed_fft))
    threshold_B = threshold_percentage * np.max(np.abs(B_windowed_fft))

    A_windowed_fft[np.abs(A_windowed_fft) < threshold_A] = 0
    B_windowed_fft[np.abs(B_windowed_fft) < threshold_B] = 0

    freqs = fftfreq(N, dt)

    A_theta = np.arctan2(np.imag(A_windowed_fft), np.real(A_windowed_fft))
    B_theta = np.arctan2(np.imag(B_windowed_fft), np.real(B_windowed_fft))

    phase_difference = A_theta - B_theta

    plt.figure()
    plt.semilogy(freqs[:N//2], 2.0/N * np.abs(A_windowed_fft)[:N//2], label="Windowed A")
    plt.semilogy(freqs[:N//2], 2.0/N * np.abs(B_windowed_fft)[:N//2], label="Windowed B")
    plt.semilogy(freqs[:N//2], 2.0/N * np.abs(A_fft)[:N//2], label="Pure A")
    plt.semilogy(freqs[:N//2], 2.0/N * np.abs(B_fft)[:N//2], label="Pure B")
    plt.xlabel("Frequency (1/fs)")
    plt.ylabel("Magnitude")
    plt.title("FFT of Properties A and B")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(freqs[:N//2], phase_difference[:N//2])
    plt.xlabel("Frequency (1/fs)")
    plt.ylabel("Phase Difference (radians)")
    plt.title("Phase Difference between Properties A and B")
    plt.show()

    

    




