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


import numpy as np

import libs.simulation as sim
from libs.helpers import calc_azimuth, calc_azimuth_sinc, calc_delay


rng = np.random.Generator(np.random.PCG64(123))


def simulate_sampling_freq(mic_array_pos, angles, c, snr, distance=2.2):
    # Calculate error DOA, while varying the sampling frequency
    #     interpolation factor: 100
    #     margin: 0.05 seconds
    #     distance: 220 cm
    #     angles: 0, 30, 60, 90, 120, 150, 180 degrees
    #     sampling rate: 2000 - 96000, steps of 2000 Hz

    fs = 192000

    x = distance * np.cos(angles)
    y = distance * np.sin(angles)

    loudspeaker_pos = [[x, y, 1] for (x, y) in zip(x, y)]
    sampling_rates = fs // np.array([48, 24, 20, 16, 15, 12, 10, 8, 6, 5, 4,
                                     3, 2, 1])

    total_doa_err = np.zeros((4, len(loudspeaker_pos), sampling_rates.shape[0]))

    print("Starting first experiment: varying sampling rate.")
    for i, fs in enumerate(sampling_rates):
        print(f"    Sampling rate: {fs} Hz")
        for j, position in enumerate(loudspeaker_pos):
            delay = calc_delay(mic_array_pos, position)
            rirs = sim.simulate_direct_sound(delay, fs, 20, 10, c)
            noise = rng.standard_normal(rirs.shape) / (10**(snr / 20))
            rirs += noise

            azi_no_int = calc_azimuth(rirs, mic_array_pos, fs, c)
            azi_parabolic = calc_azimuth(rirs, mic_array_pos, fs, c, "parabolic")
            azi_gaussian = calc_azimuth(rirs, mic_array_pos, fs, c, "gaussian")
            azi_sinc = calc_azimuth_sinc(rirs, mic_array_pos, fs, c, 100, 0.05)

            doa_err = abs(([azi_no_int, azi_parabolic, azi_gaussian, azi_sinc]
                           - (angles[j] * 180 / np.pi)))
            
            total_doa_err[:, j, i] = (doa_err + 180) % 360 - 180

    # Write output to npy file together with the sampling rates
    np.savez("doa_error_fs.npz", doa_err=total_doa_err, sampling_rates=sampling_rates)


def simulate_distances(mic_array_pos, angles, c, snr, fs=4000):
    # Calculate error DOA, while varying the distance
    #     interpolation factor: 100
    #     margin: 0.05 seconds
    #     distance: [50cm - 15m, steps of 50cm]
    #     angles: 0, 30, 60, 90, 120, 150, 180 degrees
    #     sampling rate: 4000 Hz
    distances = np.arange(0.5, 15, 0.5)

    print("Starting second experiment: varying loudspeaker distances.")

    total_doa_err = np.zeros((4, angles.shape[0], distances.shape[0]))

    for i, distance in enumerate(distances):
        print(f"    Distance: {distance} meters.")
        x = distance * np.cos(angles)
        y = distance * np.sin(angles)

        loudspeaker_pos = [[x, y, 1] for (x, y) in zip(x, y)]

        for j, position in enumerate(loudspeaker_pos):
            delay = calc_delay(mic_array_pos, position)
            rirs = sim.simulate_direct_sound(delay, fs, 20, 10, c)
            rirs += rng.standard_normal(rirs.shape) / (10**(snr / 20))

            azi_no_int = calc_azimuth(rirs, mic_array_pos, fs, c)
            azi_parabolic = calc_azimuth(rirs, mic_array_pos, fs, c,
                                         "parabolic")
            azi_gaussian = calc_azimuth(rirs, mic_array_pos, fs, c, "gaussian")
            azi_sinc = calc_azimuth_sinc(rirs, mic_array_pos, fs, c, 100, 0.05)

            doa_err = abs(([azi_no_int, azi_parabolic, azi_gaussian, azi_sinc]
                           - (angles[j] * 180 / np.pi)))

            total_doa_err[:, j, i] = (doa_err + 180) % 360 - 180

    # Write output to npy file together with the distances
    np.savez("doa_error_distance.npz", doa_err=total_doa_err, distances=distances)


def simulate_interpolation(mic_array_pos, angles, c, snr, fs=4000, distance=2.2):
    # Calculate error DOA, while varying the interpolation factor
    #     interpolation factor: [1 - 200], steps of 1
    #     margin: 0.05 seconds
    #     distance: 220cm
    #     angles: 0, 30, 60, 90, 120, 150, 180 degrees
    #     sampling rate: 4000 Hz
    x = distance * np.cos(angles)
    y = distance * np.sin(angles)

    loudspeaker_pos = [[x, y, 1] for (x, y) in zip(x, y)]

    interp_factors = np.arange(1, 201, 5)
    total_doa_err = np.zeros((4, len(loudspeaker_pos), interp_factors.shape[0]))

    print("Starting third experiment: varying interpolation factor.")
    for j, position in enumerate(loudspeaker_pos):
        print(f"    Loudspeaker position: {j}")
        delay = calc_delay(mic_array_pos, position)
        rirs = sim.simulate_direct_sound(delay, fs, 20, 10, c)
        rirs += rng.standard_normal(rirs.shape) / (10**(snr / 20))

        azi_no_int = calc_azimuth(rirs, mic_array_pos, fs, c)
        azi_parabolic = calc_azimuth(rirs, mic_array_pos, fs, c, "parabolic")
        azi_gaussian = calc_azimuth(rirs, mic_array_pos, fs, c, "gaussian")

        for i, interp_factor in enumerate(interp_factors):
            print(f"        Interpolation factor: {interp_factor}")
            azi_sinc = calc_azimuth_sinc(rirs, mic_array_pos, fs, c,
                                         interp_factor, 0.05)

            doa_err = abs(([azi_no_int, azi_parabolic, azi_gaussian, azi_sinc]
                           - (angles[j] * 180 / np.pi)))
            doa_err = (doa_err + 180) % 360 - 180
            total_doa_err[:, j, i] = doa_err

    # Write output to npy file together with the interpolation factors
    np.savez("doa_error_interp.npz", doa_err=total_doa_err, interp_factors=interp_factors)


def simulate_snr(mic_array_pos, angles, c, snr, fs=4000, distance=2.2):
    # Calculate error DOA, while varying the SNR
    #     interpolation factor: 100
    #     margin: 0.05 seconds
    #     distance: 220cm
    #     angles: 0, 30, 60, 90, 120, 150, 180 degrees
    #     sampling rate: 4000 Hz
    #     noise level SnR (dB): 5dB - 80dB (steps of +5dB)
    x = distance * np.cos(angles)
    y = distance * np.sin(angles)
    loudspeaker_pos = [[x, y, 1] for (x, y) in zip(x, y)]

    snr_factors = np.arange(10, 125, 5)
    total_doa_err = np.zeros((4, len(loudspeaker_pos), snr_factors.shape[0]))

    print("Starting fourth experiment: varying SNR factor.")
    for i, SNR in enumerate(snr_factors):
        print(f"    SNR Level: {SNR} dB")
        for j, position in enumerate(loudspeaker_pos):
            delay = calc_delay(mic_array_pos, position)
            rirs = sim.simulate_direct_sound(delay, fs, 20, 10, c)
            noise = rng.standard_normal(rirs.shape) / (10**(SNR / 20))
            rirs += noise

            azi_no_int = calc_azimuth(rirs, mic_array_pos, fs, c)
            azi_parabolic = calc_azimuth(rirs, mic_array_pos, fs, c,
                                         "parabolic")
            azi_gaussian = calc_azimuth(rirs, mic_array_pos, fs, c, "gaussian")
            azi_sinc = calc_azimuth_sinc(rirs, mic_array_pos, fs, c, 100, 0.05)

            doa_err = abs(([azi_no_int, azi_parabolic, azi_gaussian, azi_sinc]
                           - (angles[j] * 180 / np.pi)))
            
            total_doa_err[:, j, i] = (doa_err + 180) % 360 - 180

    # Write output to npy file together with the SNR factors
    np.savez("doa_error_snr.npz", doa_err=total_doa_err, snr_factors=snr_factors)


# Main
# Define simulated signals
T = 20  # degrees Celcius
c = sim.speed_of_sound(T)  # Speed of dry sound [m/s]
FS = 4000  # Sampling frequency [Hz]
DISTANCE = 2.2  # Distance to loudspeaker [m]
SNR = 40  # Signal to noise ratio [dB]

mic_pos = np.array([0, 0, 0])
mic_offsets = 0.025 * np.array([[1, 0, 0],
                                [-1, 0, 0],
                                [0, 1, 0],
                                [0, -1, 0],
                                [0, 0, 1],
                                [0, 0, -1]])
mic_array_pos = mic_pos + mic_offsets

# Define the angles in radians
angles = np.pi * np.arange(0, -1.1, -1/6)

# Perform first experiment
simulate_sampling_freq(mic_array_pos, angles, c, SNR, DISTANCE)
# Perform second experiment
simulate_distances(mic_array_pos, angles, c, SNR, FS)
# Perform third experiment
simulate_interpolation(mic_array_pos, angles, c, SNR, FS, DISTANCE)
# Perform fourth experiment
simulate_snr(mic_array_pos, angles, c, SNR, FS, DISTANCE)
