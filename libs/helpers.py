import itertools
import numpy as np

from audiodsp.core import localization as loc
from libs.interpolation import cc_sinc_interp


def calc_delay(mic_array_pos, loudspeaker_pos):
    return np.sqrt(np.sum((mic_array_pos - loudspeaker_pos)**2, axis=1))


def calc_azimuth(rirs, mic_array_pos, fs, c, interp="None"):
    tau = loc.calculate_tdoa(rirs, mic_array_pos, fs, c, interp=interp)
    doa = loc.calculate_doa(tau, mic_array_pos, fs, c)
    return np.arctan2(doa[1], doa[0]) * 180 / np.pi


def calc_azimuth_sinc(rirs, mic_array, fs, c, interp_factor, margin):
    max_td, _ = loc._mic_array_properties(mic_array, fs, c)
    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))
    offset = rirs.shape[0]
    tdoa_region = offset + np.arange(-max_td, max_td + 1)

    # Estimate the time difference of arrival using GCC
    r = loc.gcc(rirs[:, mic_pairs[:, 0]], rirs[:, mic_pairs[:, 1]])
    max_idices = np.argmax(np.abs(r[tdoa_region, :]), axis=0)
    tau_hat = (max_idices - max_td) / fs

    tau = cc_sinc_interp(r, tau_hat, interp_factor, fs, margin)

    impr_doa = loc.calculate_doa(tau, mic_array, fs, c)
    return np.arctan2(impr_doa[1], impr_doa[0]) * 180 / np.pi
