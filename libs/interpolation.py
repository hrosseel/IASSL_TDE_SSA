import numpy as np


def cc_sinc_interp(R: np.ndarray, tau: float, interp_mul: int, fs: int,
                   half_width: float = 0.002):
    """
    Fit a critically sampled sinc function to the maximum value of the
    cross-correlation function. Returns the improved time-delay found by the
    fitting.

    Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    interp_mul: int
        Interpolation factor equal to `T / T_i`. Where `T` is the sampling
        period of the original sampled signal. `T_i` is the interpolation
        sampling period.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    half_width: float
        interpolation half width of the sinc fitting. Specifies the maximum
        time-delay to fit the sinc funtion around the maximum of the
        cross-correlation function.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if(interp_mul <= 0):
        raise ValueError("Interpolation multiplier has to be a strictly"
                         " positive integer.")

    R = np.atleast_2d(R)
    max_ind = np.argmax(R, axis=0)

    fs_res = fs * interp_mul
    max_ind_res = max_ind * interp_mul

    # Search 10 samples around the direct path component
    n_margin_res = int(5 * interp_mul)
    search_area = np.array([d + np.arange(-n_margin_res, n_margin_res + 1)
                           for d in max_ind_res]).T / fs_res

    amplitudes = [R[idx, i] for i, idx in enumerate(max_ind)]
    cost_vector = np.zeros(search_area.shape)

    n_half_width = int(half_width * fs)
    window = np.array([idx + np.arange(-n_half_width, n_half_width + 1)
                       for idx in max_ind]).T
    t = window / fs

    for i, r in enumerate(R.T):
        for j, t_0 in enumerate(search_area[:, i]):
            cost_vector[j, i] = np.sum(np.square(np.sinc(fs * (t[:, i] - t_0))
                                       - r[window[:, i]] / amplitudes[i]))
    minima = np.argmin(cost_vector, axis=0)
    return (minima - n_margin_res) / fs_res + tau
