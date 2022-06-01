"""
    This is a snippet from the library AcoustDSP, authored by Hannes Rosseel.
    The entire library can be found here: https://github.com/hrosseel/acoustdsp

    Module which contains functions relating to the simulation of virtual
    room acoustics.

    References
    ----------
    [1] V. Valimaki and A. Haghparast, “Fractional Delay Filter Design Based
        on Truncated Lagrange Interpolation,” IEEE Signal Process. Lett.,
        vol. 14, no. 11, pp. 816–819, Nov. 2007, doi: 10.1109/LSP.2007.898856.
    [2] A. D. Pierce, Acoustics: An Introduction to Its Physical Principles
        and Applications. Cham: Springer International Publishing, 2019.
        doi: 10.1007/978-3-030-11214-1.
"""
import numpy as np
from scipy.signal import lfilter


def lagrange_fd_filter_truncated(delay: np.ndarray, N: int, K: int):
    """
    Calculate the coefficients for an N-order truncated Lagrange Fractional
    Delay (FD) FIR filter. K coefficients are removed from both each end
    of the prototype filter [1]. A trunctated FD filter has a wider
    magnitude response, at the cost of a ripple in the passband of the
    filter.

    Parameters
    ----------
    delay: np.ndarray
        An `(M, 1)` vector containing the desired fractional delays of in
        samples. For every fractional delay value, a truncated Lagrange FIR
        filter is created.
    N: int
        Specifies the filter order of the truncated Fractional Delay FIR
        filters.
    K: int
        Specifies the number of coefficients that are set to zero at each end
        of the Lagrange prototype filter.
    Returns
    -------
    filter_taps: np.ndarray
        A `(M, N + 2K)` matrix containing `M` truncated Lagrange FIR filters.
    """
    delay = np.atleast_1d(delay)
    M = N + 2*K
    filter_taps = np.zeros((M + 1, delay.shape[0]))
    for n in range(N + 1):
        filter_taps[n + K] = np.prod([(delay + M//2 - k) / (n + K - k)
                                     for k in np.arange(0, M+1) if k != n + K],
                                     axis=0)
    return filter_taps


def simulate_direct_sound(distance: np.ndarray, fs: int, N: int = 20,
                          K: int = 0, c: float = 343, ir_length: int = None,
                          scale: bool = False):
    """
        Simulate the ideal direct sound propagation measured by a microphone at
        a given distance from a sound source.

    Parameters
    ----------
    distance: np.ndarray
        Distances between a sound source and the microphone in meters,
        specified as an (M, 1) vector.
    fs: int
        Sampling frequency in Hertz.
    N: int
        Lagrange fractional delay filter order `N`. Defaults to `N = 20`.
    K: int
        Number of coefficients of the Lagrange fractional delay filter that
        are set to zero at each end of the Lagrange prototype filter.
    ir_length: int, optional
        Define the length of the output signals in samples.
    scale: bool, optional
        When set to True, scale the direct path according to the inverse
        square law.
    Returns
    -------
    signals: np.ndarray
        Return M signals of length `ir_length` || `(distance / c + 1) * fs`,
        which represent the direct sound propagation between a sound source
        and a microphone.
    """
    num_signals = distance.shape[0]

    if ir_length:
        signals = np.zeros((ir_length, num_signals))
    else:
        signals = np.array([np.zeros(np.round(distance[i] / c + 1).astype(int)
                                     * fs) for i in range(num_signals)]).T

    for idx in range(num_signals):
        # Add one second to the total duration
        delay = distance[idx] / c * fs
        # Get integer and fractional part of the delay in samples
        i_delay = int(delay // 1)
        f_delay = delay - i_delay

        if (i_delay + 1 > signals.shape[0]):
            raise IndexError("Requested delay is higher than the "
                             "input length.")

        # Dirac delta function at integer delay point
        signals[i_delay, idx] = (1 / distance[idx] ** 2) if scale else 1
        # Filter dirac delta function with a fractional delay FIR filter
        filter_taps = lagrange_fd_filter_truncated(f_delay, N, K)
        # Account for FIR filter delay
        filter_delay = filter_taps.shape[0] // 2
        signals[:-filter_delay, idx] = lfilter(filter_taps.squeeze(), 1,
                                               signals[:, idx])[filter_delay:]
    return signals


def speed_of_sound(temperature: float) -> float:
    """
    Calculate the propagation speed of dry air at a specific temperature,
    expressed in degrees Celcius [2]. Note that this calculation only
    approximates the actual propagation speed.

    Parameters
    ----------
    temperature: float
        The temperature in degrees Celcius.
    Returns
    -------
    c: float
        Speed of sound constant in meters per second.
    """
    return 331 + 0.6 * temperature
