# Copyright (C) 2021 Hannes Rosseel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from os import path

import numpy as np
import pandas
import soundfile as sf
from scipy import signal as sig

from acoustdsp.core import simulation as sim
from libs.helpers import calc_azimuth, calc_azimuth_sinc


def simulate_sampling_freq(basepath, mic_array_pos, angles, c):
    # Calculate error DOA, while varying the sampling frequency
    #     interpolation factor: 100
    #     margin: 0.05 seconds
    #     distance: 220 cm
    #     angles: 0, 30, 60, 90, 120, 150, 180 degrees
    #     sampling rate: 4000 - 192000, steps of 8000 Hz
    FS = 192000
    decimations = np.array([48, 24, 20, 16, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1])
    sampling_rates = FS / decimations
    total_doa_err = np.zeros((4, decimations.shape[0]))
    print("Starting first experiment: varying sampling rate.")

    for a in angles:
        irs, fs = sf.read(path.join(basepath,
                                    f"ir_100cm_{str(abs(a))}deg.wav"))
        reference_doa = calc_azimuth(irs, mic_array_pos, fs, c)
        print(f"Angle: {str(reference_doa)} degrees.")
        for i, dec in enumerate(decimations):
            fs_dec = int(fs / dec)
            print(f"    Sampling rate: {fs_dec} Hz.")
            if (fs != fs_dec):
                rirs = sig.decimate(irs, dec, ftype="fir", axis=0)
            else:
                rirs = irs

            azi_no_int = calc_azimuth(rirs, mic_array_pos, fs_dec, c)
            azi_parabolic = calc_azimuth(rirs, mic_array_pos, fs_dec, c,
                                         "parabolic")
            azi_gaussian = calc_azimuth(rirs, mic_array_pos, fs_dec, c,
                                        "gaussian")
            azi_sinc = calc_azimuth_sinc(rirs, mic_array_pos, fs_dec, c, 100,
                                         0.05)

            doa_err = abs(([azi_no_int, azi_parabolic, azi_gaussian, azi_sinc]
                           - reference_doa))
            doa_err = (doa_err + 180) % 360 - 180
            total_doa_err[:, i] += doa_err

    # Write output to csv file
    pd = pandas.DataFrame(total_doa_err.T / len(angles))
    pd.set_index(sampling_rates / 1000).to_csv("doa_error_fs.csv")


def simulate_interpolation(basepath, mic_array_pos, angles, c):
    # Calculate error DOA, while varying the interpolation factor
    #     interpolation factor: [1 - 200], steps of 1
    #     margin: 0.05 seconds
    #     distance: 220cm
    #     angles: 0, 30, 60, 90, 120, 150, 180 degrees
    #     sampling rate: 4000 Hz
    DEC = 48

    interp_factors = np.arange(1, 201, 5)
    total_doa_err = np.zeros((4, interp_factors.shape[0]))
    print("Starting second experiment: varying interpolation factor.")
    for a in angles:
        irs, fs = sf.read(path.join(basepath,
                                    f"ir_170cm_{str(abs(a))}deg.wav"))
        reference_doa = calc_azimuth(irs, mic_array_pos, fs, c)

        fs_dec = int(fs / DEC)
        rirs = sig.decimate(irs, DEC, axis=0)

        azi_no_int = calc_azimuth(rirs, mic_array_pos, fs_dec, c)
        azi_parabolic = calc_azimuth(rirs, mic_array_pos, fs_dec, c,
                                     "parabolic")
        azi_gaussian = calc_azimuth(rirs, mic_array_pos, fs_dec, c, "gaussian")

        for i, interp_factor in enumerate(interp_factors):
            print(f"        Interpolation factor: {interp_factor}")
            azi_sinc = calc_azimuth_sinc(rirs, mic_array_pos, fs_dec, c,
                                         interp_factor, 0.05)
            doa_err = abs(([azi_no_int, azi_parabolic, azi_gaussian, azi_sinc]
                           - reference_doa))
            doa_err = (doa_err + 180) % 360 - 180
            total_doa_err[:, i] += doa_err

    pd = pandas.DataFrame(total_doa_err.T / len(angles))
    pd.set_index(interp_factors).to_csv("doa_error_interp.csv")


# Main
# Define simulated signals
T = 20  # degrees Celcius
c = sim.speed_of_sound(T)  # Speed of dry sound [m/s]

# Store results in current directory
basepath = "./anechoic_measurements"

mic_pos = np.array([0, 0, 0])
mic_offsets = np.array([[0.025, 0, 0],
                        [-0.025, 0, 0],
                        [0, 0.025, 0],
                        [0, -0.025, 0],
                        [0, 0, 0.025],
                        [0, 0, -0.025]])
mic_array_pos = mic_pos + mic_offsets

# Define the angles between source and receiver in radians
angles = np.array([0, -30, -60, -90, -120, -150, -180])

# Perform the first experiment
simulate_sampling_freq(basepath, mic_array_pos, angles, c)
# Perform the second experiment
simulate_interpolation(basepath, mic_array_pos, angles, c)
