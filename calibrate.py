# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import shift
from features import *
from radar import *
import argparse

# Converts an array of target locations (N x 2) into a binary polar image
def targets_to_polar_image(targets, shape):
    polar = np.zeros(shape)
    N = targets.shape[0]
    for i in range(0, N):
        polar[targets[i, 0], targets[i, 1]] = 255
    return polar

# Returns a 3 x 3 rotation matrix for a given theta (yaw about the z axis)
def get_rotation(theta):
    R = np.identity(3)
    R[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    return R

# Returns a 3 x N array of lidar points (x, y, z)
# def load_lidar(filename):
#     lidar = np.loadtxt(filename, delimiter=',')
#     return lidar[:, 0:3].transpose()

def load_lidar(path):
    points = np.fromfile(path, dtype=np.float32).reshape((-1, 6))
    return points[:, :3].T

# Converts a lidar point cloud (3 x N) into a top-down cartesian image
def lidar_to_cartesian_image(pc, cart_pixel_width, cart_resolution):
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    cart_img = np.zeros((cart_pixel_width, cart_pixel_width))
    for i in range(0, pc.shape[1]):
        if pc[2, i] < -1.5 or pc[2, i] > 1.5:
            continue
        u = int((cart_min_range - pc[1, i]) / cart_resolution)
        v = int((cart_min_range - pc[0, i]) / cart_resolution)
        if 0 < u and u < cart_pixel_width and 0 < v and v < cart_pixel_width:
            cart_img[v, u] = 255
    return cart_img

# Converts lidar point cloud (3 x N) into a top-down polar image (azimuth x range)
def lidar_to_polar_image(pc, range_resolution, azimuth_resolution, range_bins, azimuth_bins):
    polar = np.zeros((azimuth_bins, range_bins))
    for i in range(0, pc.shape[1]):
        if x[2, i] < -1.5 or x[2, i] > 1.5:
            continue
        r = np.sqrt(x[0, i]**2 + x[1, i]**2)
        theta = np.arctan2(x[1, i], x[0, i])
        if theta < 0:
            theta += 2 * np.pi
        range_bin = int(r / range_resolution)
        azimuth_bin = int(theta / azimuth_resolution)
        if 0 < range_bin and range_bin < range_bins and 0 < azimuth_bin and azimuth_bin < azimuth_bins:
            polar[azimuth_bin, range_bin] = 255
    polar = np.flip(polar, axis=0)
    return polar

def get_closest_frame(query_time, target_times, targets):
    times = np.array(target_times)
    closest = np.argmin(np.abs(times - query_time))
    assert(np.abs(query_time - times[closest]) < 1.0), "closest time to query: {} in rostimes not found.".format(query_time)
    return targets[closest]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='path to /lidar and /radar')
    parser.add_argument('--lightmode', action='store_true', help='use lightmode when making radar-to-lidar plots')
    args = parser.parse_args()
    root = args.root
    lightmode = args.lightmode
    radar_root = osp.join(root, 'radar')
    lidar_root = osp.join(root, 'lidar')
    radar_files = sorted([f for f in os.listdir(radar_root) if 'png' in f])
    lidar_files = sorted([f for f in os.listdir(lidar_root) if 'bin' in f])
    lstamps = [float(f.split('.')[0]) * 1e-6 for f in lidar_files]
    rstamps = [float(f.split('.')[0]) * 1e-6 for f in radar_files]
    min_sample_rate = 1.0
    rstamps2 = []
    radar_files2 = []
    tprev = 0
    for i,t in enumerate(rstamps):
        if t - tprev < min_sample_rate:
            continue
        rstamps2.append(t)
        radar_files2.append(radar_files[i])
        tprev = t
    rstamps = rstamps2
    radar_files = radar_files2
    lidar_files = [get_closest_frame(rstamp, lstamps, lidar_files) for rstamp in rstamps]
    assert(len(radar_files) == len(lidar_files))

    if not osp.exists("figs"):
        os.makedirs("figs")
    min_range = 42                  # We ignore radar bins [0, min_range)
    radar_resolution = 0.0596       # Each range bin in the radar data corresponds to this many meters.
    azimuth_bins = 400
    cart_resolution = radar_resolution        # Meters per pixel in the cartesian image (set to a multiple of radar_resolution)
    max_range = 100
    max_bins = int(max_range / radar_resolution)
    cart_pixel_width = int(max_range / radar_resolution)         # Width of the cartesian image in pixels
    azimuth_step = 2 * np.pi / azimuth_bins      # Each row in the polar data corresponds to this azimuth step in radians
    calibrate_translation = False    # Set to false to skip translation estimation
    upsample_azimuths = 3           # The factor with which to upsample the azimuth axis of the polar radar data
    cart_res2 = cart_resolution     # (>= radar_resolution) decrease for better translation estimation
    cart_width2 = int(2 * max_range / cart_res2)
    if upsample_azimuths > 1.0:
        azimuth_step = azimuth_step / upsample_azimuths
    visualize_results = True

    rotations = []
    translations = []

    for i in range(0, 1):
        # Load radar data and upsample along azimuth axis
        times, azimuths, _, fft_data = load_radar(osp.join(radar_root, radar_files[i]), fix_azimuths=True)
        print(azimuths.shape, azimuths.dtype)
        plt.scatter(list(range(400)), azimuths)
        plt.show()
        fft_data = fft_data[:, :max_bins]
        azimuth_bins = fft_data.shape[0]
        range_bins = fft_data.shape[1]
        if upsample_azimuths > 1.0:
            azimuths[1] = azimuth_step
            fft_data = cv2.resize(fft_data, dsize = (0, 0), fx = 1, fy = upsample_azimuths, interpolation = cv2.INTER_CUBIC)

        # Extract radar target locations and convert these into polar and cartesian images
        targets = cen2018features(fft_data)
        polar = targets_to_polar_image(targets, fft_data.shape)
        cart = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_resolution, cart_pixel_width, fix_wobble=False)
        cart = np.where(cart > 0, 255, 0)

        # Load lidar data and convert it into polar and cartesian images
        x = load_lidar(osp.join(lidar_root, lidar_files[i]))
        cart_lidar = lidar_to_cartesian_image(x, cart_pixel_width, cart_resolution)
        polar_lidar = lidar_to_polar_image(x, radar_resolution, azimuth_step, range_bins, azimuth_bins * upsample_azimuths)

        # Estimate the rotation using the Fourier Mellin transform on the polar images
        f1 = np.fft.fft2(polar)
        f2 = np.fft.fft2(polar_lidar)
        p = (f2 * f1.conjugate())
        p = p / abs(p)
        p = np.fft.ifft2(p)
        p = abs(p)
        rotation_index = np.where(p == np.amax(p))[0][0]
        rotation = rotation_index * azimuth_step
        if rotation > np.pi:
            rotation = 2 * np.pi - rotation
        print('rotation index: {} rotation: {} radians, {} degrees'.format(rotation_index, rotation, rotation * 180 / np.pi))
        rotations.append(rotation)
        R = get_rotation(rotation) # Rotation to convert points in lidar frame to points in radar frame (R_12)

        # Rotate the lidar scan such that only the translation offset remains
        xprime = x
        for j in range(0, x.shape[1]):
            xprime[:,j] = np.squeeze(np.matmul(R, x[:,j].reshape(3,1)))

        # Estimate the translation using the Fourier Mellin transform on the cartesian images
        if calibrate_translation:
            cart1 = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_res2, cart_width2)
            cart2 = lidar_to_cartesian_image(xprime, cart_width2, cart_res2)
            f1 = np.fft.fft2(cart1)
            f2 = np.fft.fft2(cart2)
            p = (f2 * f1.conjugate())
            p = p / abs(p)
            p = np.fft.ifft2(p)
            p = abs(p)
            delta_x = np.where(p == np.amax(p))[0][0]
            delta_y = np.where(p == np.amax(p))[1][0]
            if delta_x > cart_width2 / 2:
                delta_x -= cart_width2
            if delta_y > cart_width2 / 2:
                delta_y -= cart_width2
            delta_x *= cart_res2
            delta_y *= cart_res2
            xbar = np.array([delta_x, delta_y, 1]).reshape(3, 1)
            print('delta_x: {} delta_y: {}'.format(xbar[0], xbar[1]))
            translations.append(xbar.transpose())

    rotations = np.array(rotations)
    rotation = np.mean(rotations)
    print('rotation: {} radians, {} degrees'.format(rotation, rotation * 180 / np.pi))
    if calibrate_translation:
        translations = np.array(translations)
        translation = np.mean(translations, axis=0)
        print('x: {} y : {}'.format(translation[0, 0], translation[0, 1]))

    if visualize_results:
        cart_resolution = 0.2384
        cart_pixel_width = 838
        azimuth_step = np.pi / 200
        azimuth_bins = 400
        R = get_rotation(rotation)
        print(rotations)
        plt.plot(np.array(range(rotations.shape[0])), rotations)
        plt.show()
        for i in range(0, len(radar_files)):
            _, azimuths, _, fft_data = load_radar(osp.join(radar_root, radar_files[i]))
            targets = cen2018features(fft_data)
            polar = targets_to_polar_image(targets, fft_data.shape)
            cart = radar_polar_to_cartesian(azimuths, polar, radar_resolution, cart_resolution, cart_pixel_width)
            cart = np.where(cart > 0, 255, 0)
            x = load_lidar(osp.join(lidar_root, lidar_files[i]))
            cart_lidar = lidar_to_cartesian_image(x, cart_pixel_width, cart_resolution)
            for j in range(0, x.shape[1]):
                x[:,j] = np.squeeze(np.matmul(R, x[:,j].reshape(3,1)))
            cart_lidar2 = lidar_to_cartesian_image(x, cart_pixel_width, cart_resolution)
            if lightmode:
                rgb = np.ones((cart_pixel_width, cart_pixel_width, 3), np.uint8) * 255
                mask = np.logical_not(cart_lidar2 == 255) * 255
                rgb[..., 1] = mask
                rgb[..., 2] = mask
                mask2 = np.logical_not(cart == 255) * 255
                rgb[..., 0] = np.logical_or(cart_lidar2, mask2) * 255
                rgb[..., 1] = np.logical_and(rgb[..., 1], mask2) * 255
                rgb[..., 2] = np.logical_or(rgb[..., 2], cart) * 255
                rgb[..., 0] *= np.logical_not(np.logical_and(cart_lidar2, cart))
            else:
                rgb = np.zeros((cart_pixel_width, cart_pixel_width, 3), np.uint8)
                rgb[..., 0] = cart_lidar2
                rgb[..., 1] = cart

            cv2.imwrite(osp.join("figs", "combined" + str(i) + ".png"), np.flip(rgb, axis=2))
            cv2.imwrite(osp.join("figs", "combined" + str(i) + ".png"), np.flip(rgb, axis=2))
            fig, axs = plt.subplots(1, 3, tight_layout=True)
            axs[0].imshow(cart, cmap=cm.gray)
            axs[1].imshow(cart_lidar, cmap=cm.gray)
            axs[2].imshow(rgb)
            plt.show()
