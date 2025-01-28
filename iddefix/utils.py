import numpy as np
from scipy.constants import c as c_light

def pars_to_dict(pars):
    """Converts a list of parameters into a dictionary of parameter groups.

    This function takes a list of parameters `pars` and groups them into
    dictionaries of three parameters (e.g. Rs, Q, resonant_frequency) each. 
    The keys of the resulting dictionary are integers starting from 0, 
    and the values are lists containing three consecutive parameters from 
    the input list.

    Args:
        pars: A list or array of parameters to be grouped.

    Returns:
        dict: A dictionary where keys are integers and values are
             lists of three parameters.

    Raises:
        ValueError: If the length of `pars` is not a multiple of 3.
    """

    if len(pars) % 3 != 0:
        raise ValueError("Input list length must be a multiple of 3")

    grouped_parameters = {}
    for i in range(0, len(pars), 3):
        grouped_parameters[i // 3] = pars[i : i + 3]

    return grouped_parameters


def compute_fft(data_time, data_wake_potential, fmax=3e9, samples=1001):
    """
    Compute the Fourier transform of a wake potential and return the frequencies 
    and impedance values within a specified frequency range.

    Parameters
    ----------
    data_time : array-like
        Array of time values (in nanoseconds) corresponding to the wake potential data.
    data_wake_potential : array-like
        Array of wake potential values corresponding to `data_time`.
    fmax : float, optional
        Maximum frequency (in Hz) to include in the output. Defaults to 3e9 Hz (3 GHz).
    samples : int, optional
        Number of samples to determine the resolution of the Fourier transform. Defaults to 1001.

    Returns
    -------
    f : ndarray
        Array of frequency values (in Hz) within the range [0, `fmax`).
    Z : ndarray
        Array of impedance values corresponding to the frequencies in `f`.

    Notes
    -----
    - The time array (`data_time`) is assumed to be evenly spaced.
    - The spatial sampling interval `ds` is computed based on the time step and 
      the speed of light in vacuum.
    - The Fourier transform is computed using the `numpy.fft` module, and the 
      results are normalized by the sampling interval (`ds`).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.constants import c
    >>> time = np.linspace(0, 10, 100)  # Time in nanoseconds
    >>> wake_potential = np.sin(2 * np.pi * 1e9 * time * 1e-9)  # Example wake potential
    >>> f, Z = compute_fft(time, wake_potential, fmax=2e9, samples=500)
    >>> print(f.shape, Z.shape)
    (500, 500)
    """
    
    ds = (data_time[1] - data_time[0])* 1e-9 * c_light
    N = int((c_light/ds)//fmax*samples)
    Z = np.fft.fft(data_wake_potential, n=N)
    f = np.fft.fftfreq(len(Z), ds/c_light)

    # Mask invalid frequencies
    mask  = np.logical_and(f >= 0 , f < fmax)
    Z = Z[mask]*ds
    f = f[mask]                                  

    return f, Z