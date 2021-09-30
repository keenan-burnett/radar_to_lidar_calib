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

def get_azimuth_index(azms, aquery):
    closest = np.argmin(np.abs(azms - aquery))
    if azms[closest] < aquery:
        if closest < azms.shape[0] - 1:
            if azms[closest + 1] == azms[closest]:
                closest += 0.5
            elif azms[closest + 1] > azms[closest]:
                closest += (aquery - azms[closest]) / (azms[closest + 1] - azms[closest])
    elif azms[closest] > aquery:
        if closest > 0:
            if azms[closest - 1] == azms[closest]:
                closest -= 0.5
            elif azms[closest - 1] < azms[closest]:
                closest -= (azms[closest] - aquery) / (azms[closest] - azms[closest - 1])
    return closest

def fix_wobble(raw_data):
    encoder_size = 5600
    azms = (raw_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    fft_data = raw_data[:, 11:].copy().astype(np.float32)
    a_step = (azms[-1] - azms[0]) / (azms.shape[0] - 1)
    for i in range(1, azms.shape[0] - 1):
        aquery = azms[0] + a_step * i
        aindex = get_azimuth_index(azms, aquery)
        raw_data[i, 11:] = bilinear_intep(fft_data, aindex).astype(np.uint8)
        raw_data[i, 8:10] = convert_to_byte_array(np.uint16(encoder_size * aquery / (2 * np.pi)), d=16)

def load_radar(example_path, fix_wob=False):
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
    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    if fix_wob:
        fix_wobble(raw_example_data)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.
    fft_data = np.squeeze(fft_data)
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
    # This fixes the wobble in the old CIR204 data from Boreas (keenan)
    if fix_wobble and radar_resolution == 0.0596:
        azimuths = azimuths.reshape((1, 1, 400))  # 1 x 1 x 400
        sample_angle = np.expand_dims(sample_angle, axis=-1)  # H x W x 1
        diff = np.abs(azimuths - sample_angle)
        c3 = np.argmin(diff, axis=2)
        azimuths = azimuths.squeeze()
        c3 = c3.reshape(cart_pixel_width, cart_pixel_width)  # azimuth indices (closest)
        mindiff = sample_angle.squeeze() - azimuths[c3]
        sample_angle = sample_angle.squeeze()
        mindiff = mindiff.squeeze()

        subc3 = c3 * (c3 < 399)
        aplus = azimuths[subc3 + 1]
        a1 = azimuths[subc3]
        delta1 = mindiff * (mindiff > 0) * (c3 < 399) / (aplus - a1)
        subc3 = c3 * (c3 > 0)
        a2 = azimuths[subc3]
        aminus = azimuths[1 + (c3 > 0) * (subc3 - 2)]
        delta2 = mindiff * (mindiff < 0) * (c3 > 0) / (a2 - aminus)
        sample_v = c3 + delta1 + delta2
        sample_v = sample_v.astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
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
