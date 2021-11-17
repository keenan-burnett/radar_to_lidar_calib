################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################

from typing import AnyStr, Tuple
import numpy as np
import cv2

CTS350 = 0
CIR204 = 1

def load_radar(example_path, fix_azimuths=False):
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
    """
    # Hard coded configuration to simplify parsing code
    encoder_size = 5600
    #t = float(example_path.split('.')[0]) * 1.0e-6
    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    N = raw_example_data.shape[0]
    azimuth_step = 2 * np.pi / N
    if fix_azimuths:
    	azimuths = np.zeros((N, 1), dtype=np.float32)
    	for i in range(N):
            azimuths[i, 0] = i * azimuth_step
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32) / 255.
    return timestamps, azimuths, valid, fft_data

def radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                             interpolate_crossover=True, fix_wobble=True):
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    M = azimuths.shape[0]
    azms = azimuths.squeeze()
    if fix_wobble:
        c3 = np.searchsorted(azms, sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azms[c3]
        diff = sample_angle.squeeze() - a3
        a2 = azms[c2]
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u.astype(np.float32), sample_v.astype(np.float32)), -1)
    print(polar_to_cart_warp.dtype)
    print(fft_data.dtype)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)

def cartesian_to_polar(cart: np.ndarray, radial_step: float, azimuth_step : float, radial_bins: int,
        azimuth_bins: int, cart_resolution: float) -> np.ndarray:
    """Convert a cartesian image into polar form
    Args:
        Cart (np.ndarray): Cartesian image data
        radial_step (float): range resolution of the output polar image
        azimuth_step (float): azimuth resolution of the output polar image
        radial_bins (int): width of the output polar image (number of range bins)
        azimuth_bins (int): height of the output polar image
        cart_resolution (float): Cartesian resolution (metres per pixel)

    Returns:
        np.ndarray: azimuth_bins x radial_bins polar image
    """
    max_range = radial_step * radial_bins
    angles = np.linspace(0, 2 * np.pi, azimuth_bins, dtype=np.float32).reshape(azimuth_bins, 1)
    ranges = np.linspace(0, max_range, radial_bins, dtype=np.float32).reshape(1, radial_bins)
    angles = np.tile(angles, (1, radial_bins))
    ranges = np.tile(ranges, (azimuth_bins, 1))
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    cart_pixel_width = cart.shape[0]
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    u = (cart_min_range + y) / cart_resolution
    v = (cart_min_range - x) / cart_resolution
    cart_to_polar_warp = np.stack((u, v), -1)
    polar = np.expand_dims(cv2.remap(cart, cart_to_polar_warp, None, cv2.INTER_LINEAR), -1)
    return np.squeeze(polar)
