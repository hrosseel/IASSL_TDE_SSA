# Improved Acoustic Sound Source Localization By Time Delay Estimation with Subsample Accuracy
This repository contains the code files that were used to create the simulation and measurement results for the paper "Improved Acoustic Sound Source Localization By Time Delay Estimation with Subsample Accuracy", presented at the 2021 International Conference on Immersive and 3D Audio (I3DA) on September 8-10, 2021.

## Abstract
Discrete-time signal processing algorithms for Time Delay Estimation (TDE) generally yield a delay estimate that is an integer multiple of the sampling period. In applications that operate at a relatively low sampling rate or that require a highly accurate delay estimate, the TDE resolution obtained in this way may not be sufficient. One such application is 6DOF audio acquisition, in which accurate time delays are needed to estimate directions of arrival and sound source positions relative to microphone positions. Depending on the TDE algorithm and the envisaged application, several solutions have been proposed to increase the TDE accuracy, including parabolic interpolation, increasing the sampling rate, and increasing the distance between the microphones in the array. In this paper, we propose a novel method for solving the TDE resolution problem, which is directly rooted in the Nyquist-Shannon sampling theory. By fitting a continuous-time sinc function to the cross-correlation function of two measured acoustic impulses, a delay estimate can be obtained with a time resolution that is only a fraction of the sampling period. When applying this approach to a set of acoustic impulse responses measured between a single sound source and a microphone array, e.g., in a 6DOF audio acquisition scenario, the increase in TDE accuracy yields a more accurate estimate of the time differences of arrival of the source relative to the different microphones, which can eventually lead to improved source localization. A comparison of the proposed method with existing methods will be presented.

## Requirements
In order to run this code, you need to install the packages specified in [requirements.txt](requirements.txt).

The package `acoustdsp` can be found in the following [repository]().
## Cite us
In case you find this code helpful, please cite the paper once it's published in IEEE Xplore.

    Rosseel H., van Waterschoot T., "Improved Acoustic Source Localization by Time Delay Estimation with Subsample Accuracy", presented at I3DA 2021, Bologna, Italy, 2021.

## License
    Copyright (C) 2021 Hannes Rosseel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.